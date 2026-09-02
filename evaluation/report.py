"""Validation and Markdown rendering for seasonal evaluation evidence."""

from evaluation.runner import CONDITIONS, MODEL_NAMES, SEASON_NAMES


METRIC_SETS = ("external_five_class", "heldout_six_class")


def present_metric_sets(results):
    """The metric sets every run carries, in canonical order.

    The two halves of the evidence are produced by separate long-running jobs.
    Reporting one half while the other is still computing beats reporting
    nothing, so a set is usable only when every run has it -- a metric set
    present on some runs and not others would silently compare different
    populations across rows of the same table.
    """
    runs = results.get("runs", [])
    return tuple(
        name for name in METRIC_SETS
        if runs and all(name in run.get("metrics", {}) for run in runs)
    )


def validate_results(results, metric_sets=None):
    runs = results.get("runs", [])
    if len(runs) != 20:
        raise ValueError(f"Expected 20 runs, found {len(runs)}")
    expected = {
        f"{season}-{condition}-{model}"
        for season in SEASON_NAMES
        for condition in CONDITIONS
        for model in MODEL_NAMES
    }
    actual = {run.get("run_id") for run in runs}
    if actual != expected:
        raise ValueError(f"Run IDs differ: missing={sorted(expected - actual)}")
    required_sets = set(METRIC_SETS if metric_sets is None else metric_sets)
    if not required_sets:
        raise ValueError("At least one metric set is required")
    for run in runs:
        if not required_sets <= set(run.get("metrics", {})):
            raise ValueError(
                f"{run['run_id']} is missing "
                f"{sorted(required_sets - set(run.get('metrics', {})))}"
            )
        for name, metrics in run["metrics"].items():
            required = {
                "overall_accuracy", "kappa", "macro_precision", "macro_recall",
                "macro_f1", "confusion_matrix", "labels", "sample_count", "per_class",
            }
            missing = required - set(metrics)
            if missing:
                raise ValueError(f"{run['run_id']} {name} missing {sorted(missing)}")
    return results


def delta_rows(results, metric_set):
    validate_results(results, [metric_set])
    indexed = {(run["season"], run["condition"], run["model"]): run
               for run in results["runs"]}
    rows = []
    for season in SEASON_NAMES:
        for model in MODEL_NAMES:
            before = indexed[(season, "before", model)]["metrics"][metric_set]
            after = indexed[(season, "after", model)]["metrics"][metric_set]
            before_accuracy = before["overall_accuracy"]
            after_accuracy = after["overall_accuracy"]
            rows.append({
                "season": season,
                "model": model,
                "before_accuracy": before_accuracy,
                "after_accuracy": after_accuracy,
                "delta_percentage_points": (
                    (after_accuracy - before_accuracy) * 100
                    if before_accuracy is not None and after_accuracy is not None
                    else None
                ),
            })
    return rows


def restricted_accuracy(metrics, excluded_labels):
    """Overall accuracy over reference rows outside `excluded_labels`.

    Before and After do not classify the same thing: After adds Agriculture, so
    its overall accuracy is measured on a harder task and a straight OA
    comparison charges it for work Before never attempted. Dropping the excluded
    reference *rows* while keeping every prediction column fixes that -- an
    After model that answers "Agriculture" on a Forest row is still counted
    wrong, it just is not asked the Agriculture questions Before was never asked.
    """
    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    kept = [index for index, label in enumerate(labels)
            if label not in set(excluded_labels)]
    total = sum(sum(matrix[index]) for index in kept)
    correct = sum(matrix[index][index] for index in kept)
    return _safe_divide(correct, total), total


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def comparable_rows(results, excluded_labels=("Agriculture",)):
    """Before/After held-out accuracy on the class inventory both share."""
    validate_results(results, ["heldout_six_class"])
    indexed = {(run["season"], run["condition"], run["model"]): run
               for run in results["runs"]}
    rows = []
    for season in SEASON_NAMES:
        for model in MODEL_NAMES:
            values = {}
            for condition in CONDITIONS:
                metrics = indexed[(season, condition, model)]["metrics"][
                    "heldout_six_class"]
                accuracy, support = restricted_accuracy(metrics, excluded_labels)
                values[condition] = accuracy
                values[f"{condition}_support"] = support
            delta = (
                (values["after"] - values["before"]) * 100
                if values["before"] is not None and values["after"] is not None
                else None
            )
            rows.append({
                "season": season, "model": model,
                "before_accuracy": values["before"],
                "after_accuracy": values["after"],
                "before_support": values["before_support"],
                "after_support": values["after_support"],
                "delta_percentage_points": delta,
            })
    return rows


def class_recall_rows(results, class_name):
    """Per-model recall for one class, Before against After."""
    validate_results(results, ["heldout_six_class"])
    indexed = {(run["season"], run["condition"], run["model"]): run
               for run in results["runs"]}
    rows = []
    for season in SEASON_NAMES:
        for model in MODEL_NAMES:
            entry = {"season": season, "model": model}
            for condition in CONDITIONS:
                metrics = indexed[(season, condition, model)]["metrics"][
                    "heldout_six_class"]
                per_class = metrics["per_class"].get(class_name, {})
                entry[condition] = per_class.get("recall")
                entry[f"{condition}_support"] = per_class.get("support")
            rows.append(entry)
    return rows


def _percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_markdown(results):
    present = present_metric_sets(results)
    validate_results(results, present)
    lines = [
        "# Seasonal six-class Madhya Pradesh accuracy report",
        "",
        "This report measures agreement with high-confidence public-map consensus; "
        "it is not field-survey ground truth. External five-class results use Forest, "
        "Water, Buildings, Bare, and Agriculture. Bare-ground project classes are "
        "collapsed to Bare only for that external comparison: Before's Soil and Sand "
        "both read as Bare, After's Barren Land reads as Bare. Held-out results "
        "retain each condition's own project labels.",
        "",
    ]
    if "external_five_class" not in present:
        lines.extend([
            "> The external five-class half of the evidence is still computing and "
            "is omitted here rather than partially reported.",
            "",
        ])
    for metric_set, title in (
        ("external_five_class", "External five-class"),
        ("heldout_six_class", "Held-out six-class"),
    ):
        if metric_set not in present:
            continue
        lines.extend([f"## {title}", ""])
        for season in SEASON_NAMES:
            lines.extend([
                f"### {season.title()}",
                "",
                "| Model | Before OA | After OA | Delta (pp) |",
                "|---|---:|---:|---:|",
            ])
            for row in delta_rows(results, metric_set):
                if row["season"] != season:
                    continue
                delta = row["delta_percentage_points"]
                lines.append(
                    f"| {row['model'].upper()} | {_percent(row['before_accuracy'])} | "
                    f"{_percent(row['after_accuracy'])} | "
                    f"{'N/A' if delta is None else f'{delta:+.2f}'} |"
                )
            lines.append("")
    if "heldout_six_class" in present:
        lines.extend([
            "## Held-out accuracy on the shared five classes",
            "",
            "Overall accuracy above is not a like-for-like comparison: After is scored "
            "on six classes where Before was scored on the four it could actually "
            "predict, so it is charged for a question Before was never asked. These "
            "figures drop the Agriculture reference rows and keep every prediction "
            "column, so an After model that answers Agriculture on a Forest row is "
            "still counted wrong.",
            "",
        ])
        for season in SEASON_NAMES:
            lines.extend([
                f"### {season.title()}",
                "",
                "| Model | Before OA | After OA | Delta (pp) | Rows scored |",
                "|---|---:|---:|---:|---:|",
            ])
            for row in comparable_rows(results):
                if row["season"] != season:
                    continue
                delta = row["delta_percentage_points"]
                lines.append(
                    f"| {row['model'].upper()} | {_percent(row['before_accuracy'])} | "
                    f"{_percent(row['after_accuracy'])} | "
                    f"{'N/A' if delta is None else f'{delta:+.2f}'} | "
                    f"{row['before_support']} / {row['after_support']} |"
                )
            lines.append("")

        lines.extend([
            "## Agriculture recall",
            "",
            "Before has no Agriculture rows at all, which is the point: the class was "
            "configured but its source asset carried no label, so it never reached "
            "training.",
            "",
        ])
        for season in SEASON_NAMES:
            lines.extend([
                f"### {season.title()}",
                "",
                "| Model | Before recall | After recall | After rows |",
                "|---|---:|---:|---:|",
            ])
            for row in class_recall_rows(results, "Agriculture"):
                if row["season"] != season:
                    continue
                lines.append(
                    f"| {row['model'].upper()} | {_percent(row['before'])} | "
                    f"{_percent(row['after'])} | {row['after_support']} |"
                )
            lines.append("")

    lines.extend([
        "## Reference limitations",
        "",
        "WorldCover, WorldCereal (2021), and GHSL (2018) are temporally mismatched "
        "with the 2025 Sentinel composites. Low-confidence or conflicting pixels are "
        "masked.",
        "",
    ])
    return "\n".join(lines)


def write_confusion_matrices(results, output_dir):
    """Render fixed-order confusion matrices from result JSON only."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    validate_results(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in results["runs"]:
        for metric_set in METRIC_SETS:
            metrics = run["metrics"][metric_set]
            figure, axis = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                metrics["confusion_matrix"], annot=True, fmt="g", cmap="Blues",
                xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=axis,
            )
            axis.set(xlabel="Predicted", ylabel="Reference",
                     title=f"{run['run_id']} — {metric_set}")
            figure.tight_layout()
            path = output_dir / f"{run['run_id']}_{metric_set}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            paths.append(str(path))
    return paths
