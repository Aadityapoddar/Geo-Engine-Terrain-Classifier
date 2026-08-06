import pytest

from evaluation.report import delta_rows, render_markdown, validate_results


def _metric(accuracy):
    return {
        "overall_accuracy": accuracy,
        "kappa": accuracy - 0.1,
        "macro_precision": accuracy,
        "macro_recall": accuracy,
        "macro_f1": accuracy,
        "confusion_matrix": [[1]],
        "labels": ["Only"],
        "sample_count": 1,
        "per_class": {"Only": {"precision": 1, "recall": 1, "f1": 1,
                                      "support": 1, "predicted": 1}},
    }


def _results():
    runs = []
    for season in ("winter", "summer"):
        for condition, accuracy in (("before", 0.70), ("after", 0.76)):
            for model in ("rf", "svm", "xgb", "cart", "knn"):
                runs.append({
                    "run_id": f"{season}-{condition}-{model}",
                    "season": season,
                    "condition": condition,
                    "model": model,
                    "metrics": {
                        "external_five_class": _metric(accuracy),
                        "heldout_six_class": _metric(accuracy - 0.05),
                    },
                })
    return {"schema_version": "six-class-19-band-v1", "runs": runs}


def test_result_contract_and_deltas_cover_all_season_models():
    results = _results()
    validate_results(results)
    rows = delta_rows(results, "external_five_class")

    assert len(rows) == 10
    assert rows[0]["delta_percentage_points"] == pytest.approx(6.0)


def test_markdown_contains_separate_season_and_metric_claims():
    report = render_markdown(_results())

    assert "Winter" in report and "Summer" in report
    assert "External five-class" in report
    assert "Held-out six-class" in report
    assert "Soil and Sand are collapsed to Bare" in report


def test_missing_run_is_rejected():
    results = _results()
    results["runs"].pop()
    with pytest.raises(ValueError, match="20 runs"):
        validate_results(results)

