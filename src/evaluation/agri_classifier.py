"""Agricultural operation classifier for CropOS precipitation forecasts.

Translates model predictions (probability or mm) into actionable decisions
for Thai smallholder farming operations (pesticide, fertilizer, field traffic).

Thai farming calendar note: the rainy season runs roughly May–October.
The washoff window for contact pesticides is 4 hours post-application.
Basal fertilizers (urea, DAP) are incorporated on application and tolerate
light rain; foliar sprays behave like pesticides (washoff sensitive).
"""
from __future__ import annotations

import numpy as np

# ── Category definitions ───────────────────────────────────────────────────────
# Each entry: name, mm_lo (inclusive), mm_hi (exclusive), color for display.
CATEGORIES: list[dict] = [
    {"name": "dry",           "mm_lo": 0.0,  "mm_hi": 0.1,  "color": "#2ecc71"},
    {"name": "drizzle",       "mm_lo": 0.1,  "mm_hi": 2.0,  "color": "#a9dfbf"},
    {"name": "light_rain",    "mm_lo": 2.0,  "mm_hi": 10.0, "color": "#f0e68c"},
    {"name": "moderate_rain", "mm_lo": 10.0, "mm_hi": 25.0, "color": "#f39c12"},
    {"name": "heavy_rain",    "mm_lo": 25.0, "mm_hi": 50.0, "color": "#e74c3c"},
    {"name": "extreme",       "mm_lo": 50.0, "mm_hi": float("inf"), "color": "#7b241c"},
]

CATEGORY_NAMES = [c["name"] for c in CATEGORIES]

# ── Operation safety matrix ────────────────────────────────────────────────────
# {operation: set of categories where it is SAFE to proceed}
# Conservative: when uncertain, block the operation.
_SAFE_IN: dict[str, set[str]] = {
    "pesticide_spray":   {"dry", "drizzle"},          # washoff: block at light_rain+
    "fertilizer_foliar": {"dry", "drizzle"},          # same washoff window as pesticide
    "fertilizer_basal":  {"dry", "drizzle", "light_rain"},  # incorporated; tolerates light rain
    "field_traffic":     {"dry", "drizzle", "light_rain"},  # clay soils compact at moderate+
    "harvest":           {"dry", "drizzle"},          # grain quality affected by rain
    "irrigation":        {"dry", "drizzle", "light_rain"},  # skip if moderate+ (waterlogged)
}

SUPPORTED_OPERATIONS = sorted(_SAFE_IN.keys())


def classify_mm(mm: float) -> str:
    """Classify a single mm value into an agricultural category name."""
    for cat in CATEGORIES:
        if cat["mm_lo"] <= mm < cat["mm_hi"]:
            return cat["name"]
    return "extreme"


def classify_mm_array(mm_arr: np.ndarray) -> np.ndarray:
    """Vectorised category classification for an array of mm values.

    Args:
        mm_arr: Any-shape numpy array of precipitation (mm).
    Returns:
        String array of same shape with category names.
    """
    result = np.empty(mm_arr.shape, dtype=object)
    for cat in CATEGORIES:
        mask = (mm_arr >= cat["mm_lo"]) & (mm_arr < cat["mm_hi"])
        result[mask] = cat["name"]
    # Any unassigned defaults to extreme
    result[result == None] = "extreme"  # noqa: E711
    return result


def operation_safe(operation: str, category: str) -> bool:
    """Return True if operation is safe given the forecast precipitation category.

    Use AgriClassifier.operation_safe() in application code — this function is
    provided for convenience in simple scripts.
    """
    return category in _SAFE_IN.get(operation, set())


def category_confusion_matrix(
    pred_mm: np.ndarray,
    true_mm: np.ndarray,
) -> dict[tuple[str, str], int]:
    """Build a confusion matrix over agricultural categories.

    Args:
        pred_mm: (N,) predicted mm values.
        true_mm: (N,) actual mm values.
    Returns:
        {(pred_category, true_category): count}.
    """
    pred_cats = classify_mm_array(pred_mm)
    true_cats = classify_mm_array(true_mm)
    counts: dict[tuple[str, str], int] = {}
    for p, t in zip(pred_cats.flat, true_cats.flat, strict=False):
        key = (str(p), str(t))
        counts[key] = counts.get(key, 0) + 1
    return counts


class AgriClassifier:
    """Agricultural decision engine for CropOS predictions.

    Args:
        prob_to_mm_scale: When only a probability (not mm) is available,
            the expected mm is estimated as prob * prob_to_mm_scale.
            Default 15.0 (reasonable for Thai monsoon season average ~15mm/rain-day).
        safe_categories:  Override the default SAFE_IN table (optional).
    """

    def __init__(
        self,
        prob_to_mm_scale: float = 15.0,
        safe_categories: dict[str, set[str]] | None = None,
    ) -> None:
        self.prob_to_mm_scale = prob_to_mm_scale
        self._safe_in = safe_categories if safe_categories is not None else _SAFE_IN

    def classify_mm(self, mm: float) -> str:
        return classify_mm(mm)

    def operation_safe(self, operation: str, category: str) -> bool:
        return category in self._safe_in.get(operation, set())

    def decide_from_prob(self, prob: float, operation: str) -> dict:
        """Convert a rain probability to an agricultural go/no-go decision.

        Uses a simple expected-mm estimate: E[mm] = prob * prob_to_mm_scale.
        This is a rough heuristic; replace with regression head output when available.

        Returns dict with:
          category: agricultural category string
          expected_mm: estimated mm
          safe: bool
          message: human-readable advice
        """
        expected_mm = prob * self.prob_to_mm_scale
        category = classify_mm(expected_mm)
        safe = self.operation_safe(operation, category)
        advice = {
            "dry": "Proceed — no rain expected.",
            "drizzle": "Proceed — drizzle only, negligible washoff risk.",
            "light_rain": "Caution — light rain likely. Delay pesticide/foliar spray.",
            "moderate_rain": "Do not proceed — moderate rain. Delay all sprays and fertiliser.",
            "heavy_rain": "Do not proceed — heavy rain. Suspend field operations.",
            "extreme": "Do not proceed — extreme rainfall. Flood/damage risk. Emergency protocols.",
        }
        return {
            "prob": prob,
            "expected_mm": expected_mm,
            "category": category,
            "safe": safe,
            "operation": operation,
            "message": advice[category],
        }

    def batch_decisions(
        self,
        probs: np.ndarray,
        mm_pred: np.ndarray | None,
        operations: list[str],
        horizon_labels: list[str] | None = None,
    ) -> list[dict]:
        """Produce decisions for a batch of predictions.

        Args:
            probs:      (H,) probability per forecast horizon.
            mm_pred:    (H,) mm prediction per horizon (optional; uses prob_to_mm_scale if None).
            operations: List of operation strings to evaluate.
            horizon_labels: Human-readable labels (e.g. ['12h', '24h', '36h', '48h']).
        Returns:
            List of decision dicts, one per (horizon × operation) combination.
        """
        n_horizons = len(probs)
        if horizon_labels is None:
            horizon_labels = [f"{i+1}h" for i in range(n_horizons)]

        results = []
        for h_idx in range(n_horizons):
            p = float(probs[h_idx])
            mm = float(mm_pred[h_idx]) if mm_pred is not None else p * self.prob_to_mm_scale
            category = classify_mm(mm)
            for op in operations:
                safe = self.operation_safe(op, category)
                results.append({
                    "horizon": horizon_labels[h_idx],
                    "prob": p,
                    "expected_mm": mm,
                    "category": category,
                    "operation": op,
                    "safe": safe,
                })
        return results
