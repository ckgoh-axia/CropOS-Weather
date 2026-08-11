# tests/unit/test_agri_classifier.py
import numpy as np
import pytest
from src.evaluation.agri_classifier import (
    AgriClassifier,
    CATEGORIES,
    classify_mm_array,
    operation_safe,
)


def test_categories_are_ordered():
    """Category thresholds must be monotonically increasing."""
    thresholds = [c["mm_lo"] for c in CATEGORIES]
    assert thresholds == sorted(thresholds)


def test_classify_mm_dry():
    clf = AgriClassifier()
    assert clf.classify_mm(0.0) == "dry"
    assert clf.classify_mm(0.05) == "dry"


def test_classify_mm_drizzle():
    clf = AgriClassifier()
    assert clf.classify_mm(0.5) == "drizzle"
    assert clf.classify_mm(1.9) == "drizzle"


def test_classify_mm_boundaries():
    clf = AgriClassifier()
    assert clf.classify_mm(2.0) == "light_rain"
    assert clf.classify_mm(10.0) == "moderate_rain"
    assert clf.classify_mm(25.0) == "heavy_rain"
    assert clf.classify_mm(50.0) == "extreme"


def test_classify_mm_array_shape():
    arr = np.array([[0.0, 1.0, 5.0, 15.0],
                    [30.0, 60.0, 0.05, 8.0]])
    result = classify_mm_array(arr)
    assert result.shape == arr.shape
    assert result[0, 0] == "dry"
    assert result[0, 1] == "drizzle"
    assert result[0, 2] == "light_rain"
    assert result[0, 3] == "moderate_rain"
    assert result[1, 0] == "heavy_rain"
    assert result[1, 1] == "extreme"


def test_operation_safe_pesticide():
    clf = AgriClassifier()
    assert clf.operation_safe("pesticide_spray", "dry") is True
    assert clf.operation_safe("pesticide_spray", "drizzle") is True
    assert clf.operation_safe("pesticide_spray", "light_rain") is False
    assert clf.operation_safe("pesticide_spray", "moderate_rain") is False


def test_operation_safe_fertilizer():
    clf = AgriClassifier()
    assert clf.operation_safe("fertilizer_basal", "dry") is True
    assert clf.operation_safe("fertilizer_basal", "drizzle") is True
    assert clf.operation_safe("fertilizer_basal", "light_rain") is True
    assert clf.operation_safe("fertilizer_basal", "moderate_rain") is False


def test_operation_safe_field_traffic():
    clf = AgriClassifier()
    assert clf.operation_safe("field_traffic", "dry") is True
    assert clf.operation_safe("field_traffic", "light_rain") is True
    assert clf.operation_safe("field_traffic", "moderate_rain") is False


def test_agri_decision_from_prob():
    """Convert probability to mm via calibrated expectation, then classify."""
    clf = AgriClassifier(prob_to_mm_scale=15.0)  # E[mm] = prob * scale
    # High probability → moderate/heavy rain → operations restricted
    decision = clf.decide_from_prob(0.9, "pesticide_spray")
    assert decision["safe"] is False
    # Low probability → dry/drizzle → operations OK
    decision_low = clf.decide_from_prob(0.05, "pesticide_spray")
    assert decision_low["safe"] is True


def test_agri_category_confusion_matrix():
    """Given pred_mm and true_mm, build a category confusion matrix."""
    from src.evaluation.agri_classifier import category_confusion_matrix
    pred = np.array([0.05, 1.0, 12.0, 30.0])
    true = np.array([0.05, 5.0, 2.0, 55.0])  # true is higher on last two
    cm = category_confusion_matrix(pred, true)
    assert isinstance(cm, dict)
    # The model was exactly right for the first, wrong category on rest
    assert cm[("dry", "dry")] == 1
    assert cm[("drizzle", "light_rain")] == 1
