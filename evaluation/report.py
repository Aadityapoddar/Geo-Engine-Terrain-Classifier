"""Validation and Markdown rendering for seasonal evaluation evidence."""

from evaluation.runner import CONDITIONS, MODEL_NAMES, SEASON_NAMES


METRIC_SETS = ("external_five_class", "heldout_six_class")


def validate_results(results):
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
    for run in runs:
        if set(run.get("metrics", {})) != set(METRIC_SETS):
            raise ValueError(f"{run['run_id']} must contain both metric sets")
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
    validate_results(results)
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


def _percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def render_markdown(results):
    validate_results(results)
    lines = [
        "# Seasonal six-class Madhya Pradesh accuracy report",
        "",
        "This report measures agreement with high-confidence public-map consensus; "
        "it is not field-survey ground truth. External five-class results use Forest, "
        "Water, Buildings, Bare, and Agriculture. Soil and Sand are collapsed to Bare "
        "only for that external comparison. Held-out six-class results retain all six "
        "project labels.",
        "",
    ]
    for metric_set, title in (
        ("external_five_class", "External five-class"),
        ("heldout_six_class", "Held-out six-class"),
    ):
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
    lines.extend([
        "## Reference limitations",
        "",
        "WorldCover, WorldCereal (2021), and GHSL (2018) are temporally mismatched "
        "with the 2025 Sentinel composites. Low-confidence or conflicting pixels are "
        "masked. Sand has no independent external score.",
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
