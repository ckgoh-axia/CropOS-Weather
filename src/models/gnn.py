"""CropOS Heterogeneous GNN — farm-level precipitation forecasting for Thailand.

Architecture overview
---------------------
The model operates on a *heterogeneous* spatial graph with three node types:

  era5   — ERA5 reanalysis grid nodes carrying atmospheric features
  metar  — METAR airport observation nodes carrying surface weather features
  farm   — Target nodes for which we predict precipitation probabilities

Message passing proceeds in three sequential phases that mirror the physical
chain of information from global atmospheric state down to individual farms:

Phase 1 — ERA5 → METAR (global context injection)
    Each METAR node aggregates information from nearby ERA5 grid cells.  This
    primes the local METAR embeddings with large-scale atmospheric context
    before any peer-to-peer exchange happens.

Phase 2 — METAR ↔ METAR (local observational exchange)
    METAR nodes exchange information with each other over ``local_mp_steps``
    rounds of bidirectional message passing.  Relative positional offsets are
    encoded in every message so that distance and bearing between stations
    remain explicitly visible to the network.

Phase 3 — ERA5 → METAR (re-injection of global context)
    After local peer exchange, ERA5 context is injected a second time through a
    *separate* (non-weight-shared) bipartite conv layer.  This lets the model
    re-anchor the locally refined METAR embeddings to the global atmospheric
    state before they are used to decode farm-level predictions.

Decoding
--------
Farm node embeddings are updated sequentially:
    1. ERA5 → farm  (direct atmospheric context)
    2. METAR → farm (nearby surface observation context)

A sigmoid head converts the final farm embeddings to per-horizon precipitation
occurrence probabilities.  When ``dual_head=True``, a second softplus head
also returns continuous precipitation amount estimates.

Reference
---------
Architecture inspired by arXiv:2410.12938 (MIT/IBM) with adaptations for
precipitation classification over Thai agricultural regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Linear, ReLU, Sigmoid, Softplus
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing


class _RelPosBipartiteConv(MessagePassing):
    """Bipartite message-passing layer with relative positional encoding.

    Designed for directed edges between *different* node types (era5 → metar,
    era5 → farm, metar → farm).  Each message is constructed from the
    concatenation of the source features, the destination features, and the
    relative position vector (pos_src - pos_dst), so the layer is aware of
    both the feature contrast and the spatial offset along every edge.

    A residual connection adds the original destination embedding back to the
    aggregated output via ``update_mlp``, making training of deep stacks
    stable without layer normalisation.

    Parameters
    ----------
    hidden:
        Dimensionality of all node embeddings (source and destination must
        already be projected to this size before calling this layer).
    pos_dim:
        Dimensionality of the positional coordinates (default 2: lat, lon).
    dropout:
        Dropout probability applied inside both ``msg_mlp`` and ``update_mlp``.
    """

    def __init__(self, hidden: int, pos_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__(aggr="mean")

        self.msg_mlp = nn.Sequential(
            Linear(hidden * 2 + pos_dim, hidden),
            ReLU(),
            Dropout(dropout),
            Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            Dropout(dropout),
            Linear(hidden, hidden),
        )

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
        pos_src: Tensor,
        pos_dst: Tensor,
    ) -> Tensor:
        """Run bipartite message passing and return updated destination embeddings.

        Parameters
        ----------
        x_src:
            Source node features, shape ``(N_src, hidden)``.
        x_dst:
            Destination node features, shape ``(N_dst, hidden)``.
        edge_index:
            Edge index tensor, shape ``(2, E)`` with source indices in row 0
            and destination indices in row 1.
        pos_src:
            Source node positions, shape ``(N_src, pos_dim)``.
        pos_dst:
            Destination node positions, shape ``(N_dst, pos_dim)``.

        Returns
        -------
        Tensor
            Updated destination embeddings with residual, shape
            ``(N_dst, hidden)``.
        """
        aggregated = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            pos=(pos_src, pos_dst),
            size=(x_src.size(0), x_dst.size(0)),
        )
        return x_dst + self.update_mlp(torch.cat([x_dst, aggregated], dim=-1))

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        """Construct a message for each edge.

        Parameters
        ----------
        x_j:
            Source (sender) features gathered per edge, shape ``(E, hidden)``.
        x_i:
            Destination (receiver) features gathered per edge,
            shape ``(E, hidden)``.
        pos_j:
            Source positions per edge, shape ``(E, pos_dim)``.
        pos_i:
            Destination positions per edge, shape ``(E, pos_dim)``.

        Returns
        -------
        Tensor
            Per-edge messages, shape ``(E, hidden)``.
        """
        return self.msg_mlp(torch.cat([x_j, x_i, pos_j - pos_i], dim=-1))


class _RelPosSelfConv(MessagePassing):
    """Homogeneous message-passing layer with relative positional encoding.

    Designed for bidirectional edges within a single node type (metar ↔
    metar).  The interface is identical to ``_RelPosBipartiteConv`` except that
    source and destination share the same feature tensor, so there is a single
    ``x`` argument rather than ``x_src`` / ``x_dst``.

    Parameters
    ----------
    hidden:
        Dimensionality of node embeddings.
    pos_dim:
        Dimensionality of positional coordinates (default 2: lat, lon).
    dropout:
        Dropout probability inside both MLPs.
    """

    def __init__(self, hidden: int, pos_dim: int = 2, dropout: float = 0.1) -> None:
        super().__init__(aggr="mean")

        self.msg_mlp = nn.Sequential(
            Linear(hidden * 2 + pos_dim, hidden),
            ReLU(),
            Dropout(dropout),
            Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            Dropout(dropout),
            Linear(hidden, hidden),
        )

    def forward(self, x: Tensor, edge_index: Tensor, pos: Tensor) -> Tensor:
        """Run homogeneous message passing and return updated node embeddings.

        Parameters
        ----------
        x:
            Node features, shape ``(N, hidden)``.
        edge_index:
            Edge index tensor, shape ``(2, E)``.
        pos:
            Node positions, shape ``(N, pos_dim)``.

        Returns
        -------
        Tensor
            Updated node embeddings with residual, shape ``(N, hidden)``.
        """
        aggregated = self.propagate(edge_index, x=x, pos=pos)
        return x + self.update_mlp(torch.cat([x, aggregated], dim=-1))

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        """Construct a message for each edge.

        Parameters
        ----------
        x_j:
            Sender features per edge, shape ``(E, hidden)``.
        x_i:
            Receiver features per edge, shape ``(E, hidden)``.
        pos_j:
            Sender positions per edge, shape ``(E, pos_dim)``.
        pos_i:
            Receiver positions per edge, shape ``(E, pos_dim)``.

        Returns
        -------
        Tensor
            Per-edge messages, shape ``(E, hidden)``.
        """
        return self.msg_mlp(torch.cat([x_j, x_i, pos_j - pos_i], dim=-1))


class CropOSGNN(nn.Module):
    """CropOS heterogeneous GNN for farm-level precipitation forecasting.

    Implements the three-phase ERA5 → METAR → farm message-passing pipeline
    described in the module docstring.  All node types are first projected into
    a common ``hidden``-dimensional space; subsequent conv layers preserve this
    dimensionality throughout.

    Parameters
    ----------
    era5_in:
        Number of input features on ERA5 nodes.
    hidden:
        Hidden embedding dimensionality shared across all node types and layers.
    n_horizons:
        Number of forecast horizons (output width per farm node).
    local_mp_steps:
        Number of METAR ↔ METAR message-passing rounds in Phase 2.
    dropout:
        Dropout probability used in conv MLPs and the prediction head.
    metar_dropout:
        Node-level drop probability applied to METAR nodes during training
        (DropNode regularisation to prevent over-reliance on station data).
    metar_in:
        Number of input features on METAR nodes.  Defaults to
        ``len(CropOSGNN.METAR_FEATURES)`` (9) when ``None``.
    dual_head:
        If ``True``, attach a second ``Softplus`` regression head that returns
        continuous precipitation amount estimates alongside the classification
        probabilities.

    Inputs (``forward``)
    --------------------
    data : HeteroData
        A PyG heterogeneous graph with the interface described in the module
        docstring.

    Outputs
    -------
    probs : Tensor
        Shape ``(N_farm, n_horizons)``.  Precipitation occurrence probabilities
        in ``[0, 1]``.
    amounts : Tensor, optional
        Shape ``(N_farm, n_horizons)``.  Positive precipitation amount
        estimates.  Returned only when ``dual_head=True``.
    """

    METAR_FEATURES: list[str] = [
        "precip_mm",
        "rain_event",
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "alti",
        "vsby",
    ]

    def __init__(
        self,
        era5_in: int,
        hidden: int,
        n_horizons: int,
        local_mp_steps: int = 4,
        dropout: float = 0.1,
        metar_dropout: float = 0.4,
        metar_in: int | None = None,
        dual_head: bool = False,
    ) -> None:
        super().__init__()

        if metar_in is None:
            metar_in = len(self.METAR_FEATURES)

        self.metar_dropout = metar_dropout
        self.dual_head = dual_head
        self.local_mp_steps = local_mp_steps

        # Input projections — bring every node type into the shared hidden space
        self.era5_proj = nn.Linear(era5_in, hidden)
        self.metar_proj = nn.Linear(metar_in, hidden)
        self.farm_proj = nn.Linear(1, hidden)

        # Phase 1: ERA5 → METAR (initial global context injection)
        self.era5_to_metar_1 = _RelPosBipartiteConv(hidden, dropout=dropout)

        # Phase 2: METAR ↔ METAR local exchange
        self.metar_convs = nn.ModuleList(
            [_RelPosSelfConv(hidden, dropout=dropout) for _ in range(local_mp_steps)]
        )

        # Phase 3: ERA5 → METAR again (separate weights — re-inject global context)
        self.era5_to_metar_3 = _RelPosBipartiteConv(hidden, dropout=dropout)

        # Decoding: aggregate to farm nodes
        self.era5_to_farm = _RelPosBipartiteConv(hidden, dropout=dropout)
        self.metar_to_farm = _RelPosBipartiteConv(hidden, dropout=dropout)

        # Classification head — precipitation occurrence probabilities
        self.head = nn.Sequential(
            Linear(hidden, hidden // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden // 2, n_horizons),
            Sigmoid(),
        )

        # Optional regression head — precipitation amounts
        if dual_head:
            self.reg_head = nn.Sequential(
                Linear(hidden, hidden // 2),
                ReLU(),
                Dropout(dropout),
                Linear(hidden // 2, n_horizons),
                Softplus(),
            )

    def forward(
        self, data: HeteroData
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the three-phase forward pass.

        Parameters
        ----------
        data:
            Heterogeneous graph batch.  See module docstring for the expected
            node and edge attribute layout.

        Returns
        -------
        probs : Tensor
            Precipitation occurrence probabilities, shape
            ``(N_farm, n_horizons)``.
        (probs, amounts) : tuple of Tensor
            When ``self.dual_head`` is ``True``, also returns continuous amount
            estimates with the same shape.
        """
        # ------------------------------------------------------------------ #
        # 1. Project all node types into the shared hidden space              #
        # ------------------------------------------------------------------ #
        x_era5 = self.era5_proj(data["era5"].x)
        x_metar = self.metar_proj(data["metar"].x)
        x_farm = self.farm_proj(data["farm"].x)

        pos_era5 = data["era5"].pos
        pos_metar = data["metar"].pos
        pos_farm = data["farm"].pos

        # ------------------------------------------------------------------ #
        # 2. DropNode regularisation on METAR nodes (training only)          #
        # ------------------------------------------------------------------ #
        if self.training and self.metar_dropout > 0:
            keep_prob = 1.0 - self.metar_dropout
            mask = torch.bernoulli(
                torch.full(
                    (x_metar.size(0), 1),
                    keep_prob,
                    dtype=x_metar.dtype,
                    device=x_metar.device,
                )
            )
            x_metar = x_metar * mask

        # ------------------------------------------------------------------ #
        # 3. Retrieve edge indices                                            #
        # ------------------------------------------------------------------ #
        e2m = data["era5", "to", "metar"].edge_index
        m2m = data["metar", "to", "metar"].edge_index
        e2f = data["era5", "to", "farm"].edge_index
        m2f = data["metar", "to", "farm"].edge_index

        # ------------------------------------------------------------------ #
        # Phase 1: ERA5 → METAR                                              #
        # Prime METAR embeddings with large-scale atmospheric context.        #
        # ------------------------------------------------------------------ #
        x_metar = self.era5_to_metar_1(x_era5, x_metar, e2m, pos_era5, pos_metar)

        # ------------------------------------------------------------------ #
        # Phase 2: METAR ↔ METAR local exchange                              #
        # ------------------------------------------------------------------ #
        for conv in self.metar_convs:
            x_metar = conv(x_metar, m2m, pos_metar)

        # ------------------------------------------------------------------ #
        # Phase 3: ERA5 → METAR (re-inject global context)                   #
        # Separate weights from Phase 1 — allows the model to learn a        #
        # distinct "post-peer-exchange" global re-anchoring behaviour.        #
        # ------------------------------------------------------------------ #
        x_metar = self.era5_to_metar_3(x_era5, x_metar, e2m, pos_era5, pos_metar)

        # ------------------------------------------------------------------ #
        # Decode: aggregate to farm nodes                                     #
        # Sequential — METAR sees the ERA5-updated farm embedding.           #
        # ------------------------------------------------------------------ #
        x_farm = self.era5_to_farm(x_era5, x_farm, e2f, pos_era5, pos_farm)
        x_farm = self.metar_to_farm(x_metar, x_farm, m2f, pos_metar, pos_farm)

        probs = self.head(x_farm)  # (N_farm, n_horizons)

        if self.dual_head:
            return probs, self.reg_head(x_farm)
        return probs
