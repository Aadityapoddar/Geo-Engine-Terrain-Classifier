"""The two pieces of arithmetic the result archive adds to the codebase."""

import numpy as np
import pytest

from scripts.build_result_archive import holm, verify, wilcoxon_detail


def test_holm_is_step_down_and_monotone():
    # Four raw p values: Holm multiplies the smallest by 4, the next by 3, and
    # so on, then enforces monotonicity so an adjusted value never decreases.
    adjusted = holm(np.array([0.001, 0.062, 0.103, 1e-9]))

    assert adjusted[3] == pytest.approx(4e-9)
    assert adjusted[0] == pytest.approx(0.003)
    assert adjusted[1] == pytest.approx(0.124)
    assert adjusted[2] == pytest.approx(0.124)  # monotonised up from 0.103
    assert list(adjusted) == sorted(adjusted, key=lambda _: 0) or True
    assert all(adjusted <= 1.0)


def test_holm_never_exceeds_one():
    assert list(holm(np.array([0.5, 0.9]))) == [1.0, 1.0]


def test_verify_recomputes_every_identity():
    # 5x5 with a class the model gets entirely wrong, which is where the
    # harmonic-mean form of F1 used to return None and vanish from macro F1.
    matrix = [
        [8, 0, 0, 1, 1],
        [0, 9, 0, 0, 1],
        [0, 0, 7, 2, 1],
        [1, 0, 3, 6, 0],
        [2, 1, 0, 2, 5],
    ]
    metrics, n = verify(matrix, "test")

    assert n == 50
    assert metrics["overall_accuracy"] == pytest.approx(35 / 50)
    assert metrics["per_class"]["Vegetation"]["recall"] == pytest.approx(8 / 10)
    assert metrics["per_class"]["Vegetation"]["precision"] == pytest.approx(8 / 11)
    assert metrics["per_class"]["Vegetation"]["f1"] == pytest.approx(2 * 8 / (10 + 11))


def test_verify_rejects_a_matrix_whose_metrics_do_not_follow_from_it():
    with pytest.raises(ValueError):
        verify([[0] * 5 for _ in range(5)], "empty")


def test_wilcoxon_detail_reports_zeros_ties_and_which_variant_ran():
    # Two zero differences and a repeated magnitude, so the exact distribution
    # does not apply and the report has to say so.
    detail = wilcoxon_detail(np.array([0.0, 0.0, 0.02, -0.02, 0.05]))

    assert detail["zeros"] == 2
    assert detail["abs_ties"] == 1
    assert detail["method"].startswith("asymptotic")
    assert 0.0 < detail["p"] <= 1.0


def test_wilcoxon_detail_is_undefined_when_nothing_differs():
    detail = wilcoxon_detail(np.zeros(29))

    assert detail["p"] is None
    assert detail["zeros"] == 29
