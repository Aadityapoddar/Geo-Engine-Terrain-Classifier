import pytest

from evaluation.report import (
    class_recall_rows,
    comparable_rows,
    delta_rows,
    present_metric_sets,
    render_markdown,
    restricted_accuracy,
    validate_results,
)


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
            for model in ("rf", "svm", "gtb", "cart", "knn"):
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
    assert "collapsed to Bare" in report


def test_restricted_accuracy_drops_rows_but_keeps_prediction_columns():
    # Rows are reference classes, columns are predictions. The Agriculture row
    # is dropped; the Agriculture column stays, so the Forest row's single
    # Agriculture prediction is still counted as an error.
    metrics = {
        "labels": ["Forest", "Agriculture"],
        "confusion_matrix": [[3, 1], [2, 6]],
    }
    accuracy, support = restricted_accuracy(metrics, ["Agriculture"])

    assert support == 4
    assert accuracy == pytest.approx(0.75)


def test_restricted_accuracy_reports_none_for_an_absent_class():
    metrics = {"labels": ["Forest"], "confusion_matrix": [[0]]}

    assert restricted_accuracy(metrics, ["Forest"]) == (None, 0)


def test_comparable_and_recall_rows_cover_every_season_and_model():
    results = _results()

    assert len(comparable_rows(results)) == 10
    assert len(class_recall_rows(results, "Only")) == 10


def test_markdown_separates_raw_accuracy_from_the_like_for_like_view():
    report = render_markdown(_results())

    assert "Held-out accuracy on the shared five classes" in report
    assert "Agriculture recall" in report


def test_a_half_finished_metric_set_is_not_reported_as_present():
    results = _results()
    results["runs"][0]["metrics"].pop("external_five_class")

    # All-or-nothing: one run short means the whole set is unusable, because a
    # table with 19 external rows and one blank compares different populations.
    assert present_metric_sets(results) == ("heldout_six_class",)


def test_report_renders_from_the_heldout_half_alone():
    results = _results()
    for run in results["runs"]:
        run["metrics"].pop("external_five_class")

    report = render_markdown(results)

    assert "Held-out six-class" in report
    assert "still computing" in report
    assert "## External five-class" not in report


def test_validation_still_rejects_an_explicitly_required_missing_set():
    results = _results()
    for run in results["runs"]:
        run["metrics"].pop("external_five_class")

    with pytest.raises(ValueError, match="external_five_class"):
        validate_results(results, ["external_five_class"])


def test_missing_run_is_rejected():
    results = _results()
    results["runs"].pop()
    with pytest.raises(ValueError, match="20 runs"):
        validate_results(results)

