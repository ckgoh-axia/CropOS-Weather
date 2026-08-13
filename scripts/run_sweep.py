"""W&B Bayesian hyperparameter sweep for CropOSGNN.

Usage
-----
    # 1. Create the sweep (prints a sweep ID):
    python scripts/run_sweep.py --create

    # 2. Launch one or more agents (each agent runs one trial at a time):
    python scripts/run_sweep.py --agent <SWEEP_ID>

    # Or do both in one shot (runs count trials in this process):
    python scripts/run_sweep.py --create --agent --count 20

Environment variables required
-------------------------------
    WANDB_API_KEY    — your W&B API key
    WANDB_ENTITY     — your W&B user/org name (optional; omit to use default)

    Plus the normal training env vars:
    HF_TOKEN, HF_DATASET_REPO, HF_MODEL_REPO (or use --local)

Parameters swept
----------------
    history_steps     : [12, 24]          — temporal context window (hours)
    local_mp_steps    : [2, 4, 6]         — GNN message-passing rounds (paper M)
    hidden_channels   : [128, 256]        — node embedding width
    learning_rate     : log-uniform       — [5e-5, 5e-4]
    era5_to_metar_k   : [4, 8]            — ERA5→metar bipartite k-NN
    pos_weight        : [1.5, 2.5, 4.0, 6.0]  — BCE positive-class weight (miss vs FA)

Optimisation target
-------------------
    Maximise val_bss (Brier Skill Score vs climatology).
    HyperBand early termination cuts bad configs after min 5 epochs.
"""
from __future__ import annotations

import argparse
import os


SWEEP_CONFIG: dict = {
    "name": "cropos-gnn-thai-sweep",
    "method": "bayes",
    "metric": {
        "name": "bss",
        "goal": "maximize",
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 5,
        "eta": 3,
        "s": 2,
    },
    "parameters": {
        "history_steps": {
            "values": [12, 24],
        },
        "local_mp_steps": {
            "values": [2, 4, 6],
        },
        "hidden_channels": {
            "values": [128, 256],
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        "era5_to_metar_k": {
            "values": [4, 8],
        },
        "pos_weight": {
            # Controls the miss vs false-alarm tradeoff in BCE loss.
            # 1.0 → near base-rate predictions (POD ~13%); BSS ≈ 0
            # 6.0 → aggressively detects rain but inflates probabilities
            # Bayesian search finds the calibration sweet-spot automatically.
            "values": [1.5, 2.5, 4.0, 6.0],
        },
    },
}


def _sweep_train_fn(local_data_dir: str | None = None) -> None:
    """Single trial function called by the W&B agent.

    The agent does NOT call wandb.init() — the trial function must do it.
    Once init() is called the agent-populated config is available via
    wandb.config.  train() detects the active wandb.run and reuses it
    instead of calling wandb.init() again.
    """
    import wandb

    from src.training.train import train

    wandb.init()  # connect to agent-assigned run; populates wandb.config
    cfg = dict(wandb.config)
    print(f"[sweep] trial config: {cfg}", flush=True)
    train(overrides=cfg, local_data_dir=local_data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="W&B sweep runner for CropOSGNN")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new sweep and print the sweep ID.",
    )
    parser.add_argument(
        "--agent",
        metavar="SWEEP_ID",
        nargs="?",
        const="__create__",
        help=(
            "Start a sweep agent.  Pass the sweep ID from --create, or omit "
            "the value when used together with --create (ID resolved automatically)."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Maximum number of trials this agent will run (default: 20).",
    )
    parser.add_argument(
        "--local",
        metavar="DATA_DIR",
        default=None,
        help="Path to local parquet data directory (same as train.py --local).",
    )
    parser.add_argument(
        "--project",
        default="cropos-gnn-thai",
        help="W&B project name (default: cropos-gnn-thai).",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity / org name (default: WANDB_ENTITY env var).",
    )
    args = parser.parse_args()

    import wandb  # noqa: PLC0415  (deferred so --help works without wandb installed)

    sweep_id: str | None = None

    # ── create ────────────────────────────────────────────────────────────────
    if args.create:
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project=args.project,
            entity=args.entity,
        )
        print(f"Sweep created: {sweep_id}", flush=True)
        print(
            f"Dashboard: https://wandb.ai/{args.entity or '<entity>'}/{args.project}/sweeps/{sweep_id}",
            flush=True,
        )

    # ── agent ─────────────────────────────────────────────────────────────────
    if args.agent is not None:
        if args.agent == "__create__":
            # --agent was passed without a value alongside --create
            if sweep_id is None:
                parser.error("--agent without SWEEP_ID requires --create to be set too.")
            agent_sweep_id = sweep_id
        else:
            agent_sweep_id = args.agent

        local_dir = args.local

        def _trial() -> None:
            _sweep_train_fn(local_data_dir=local_dir)

        print(
            f"Starting agent for sweep {agent_sweep_id} (max {args.count} trials)...",
            flush=True,
        )
        wandb.agent(
            sweep_id=agent_sweep_id,
            function=_trial,
            count=args.count,
            project=args.project,
            entity=args.entity,
        )

    if not args.create and args.agent is None:
        parser.print_help()


if __name__ == "__main__":
    main()
