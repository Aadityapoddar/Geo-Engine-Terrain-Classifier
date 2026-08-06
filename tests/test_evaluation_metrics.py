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
