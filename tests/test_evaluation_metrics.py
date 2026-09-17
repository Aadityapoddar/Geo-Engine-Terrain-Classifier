import pytest

from evaluation.metrics import metrics_from_matrix


def test_perfect_matrix_has_perfect_summary_metrics():
    result = metrics_from_matrix([[2, 0], [0, 3]], ["A", "B"])

    assert result["overall_accuracy"] == 1.0
    assert result["kappa"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["per_class"]["A"]["f1"] == 1.0
    assert result["per_class"]["B"]["support"] == 3


def test_zero_support_class_is_explicitly_null_not_invented():
    result = metrics_from_matrix([[4, 0], [0, 0]], ["Present", "Absent"])

    assert result["per_class"]["Absent"]["recall"] is None
    assert result["per_class"]["Absent"]["f1"] is None
    assert result["macro_f1"] == pytest.approx(1.0)


def test_matrix_must_match_label_count():
    with pytest.raises(ValueError, match="square"):
        metrics_from_matrix([[1, 0]], ["A", "B"])


def test_class_the_model_got_entirely_wrong_scores_zero_not_null():
    # Reference has two of "Missed" and the model predicted two of them
    # elsewhere: precision and recall are both 0, so the harmonic mean is
    # undefined while the count form is a well-defined 0.0.
    result = metrics_from_matrix([[3, 2], [2, 0]], ["Hit", "Missed"])

    assert result["per_class"]["Missed"]["precision"] == 0.0
    assert result["per_class"]["Missed"]["recall"] == 0.0
    assert result["per_class"]["Missed"]["f1"] == 0.0
    # 2*TP/(support+predicted) for the other class, averaged with the zero.
    assert result["macro_f1"] == pytest.approx((2 * 3 / (5 + 5) + 0.0) / 2)
