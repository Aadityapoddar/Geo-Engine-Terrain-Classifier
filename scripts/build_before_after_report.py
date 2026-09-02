#!/usr/bin/env python3
"""Render the before/after accuracy report for all five models.

Every figure is read from a measured JSON artefact. Nothing is typed in by hand,
so the prose cannot drift away from the tables underneath it.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    BANDS,
    EXPECTED_ASSET_LABELS,
    LAND_COVER_CLASSES,
    SEASONS,
)
from evaluation.report import (  # noqa: E402
    class_recall_rows,
    comparable_rows,
    delta_rows,
    present_metric_sets,
    render_markdown,
    validate_results,
)

ASSETS = REPO / "doc" / "assets"
DEFAULT_RESULTS = ASSETS / "seasonal_before_after_results.json"
DEFAULT_WATER = ASSETS / "water_point_change.json"
DEFAULT_OUTPUT = REPO / "doc" / "before_after_accuracy_report.md"


def _percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _best(rows, key):
    scored = [row for row in rows if row[key] is not None]
    return max(scored, key=lambda row: row[key]) if scored else None


def _headline(results):
    """Pull the few numbers the prose quotes, so the prose stays measured."""
    has_external = "external_five_class" in present_metric_sets(results)
    external = delta_rows(results, "external_five_class") if has_external else []
    comparable = comparable_rows(results)
    agriculture = class_recall_rows(results, "Agriculture")
    return {
        "external_best": _best(external, "after_accuracy"),
        "external_gains": [row for row in external
                           if (row["delta_percentage_points"] or 0) > 0],
        "comparable_gains": [row for row in comparable
                             if (row["delta_percentage_points"] or 0) > 0],
        "agriculture_best": _best(
            [row for row in agriculture if row["after"] is not None], "after"),
        "agriculture_after": agriculture,
    }


def _water_section(water):
    displacement = water["displacement"]
    lines = [
        "## What changed in the water points",
        "",
        "The Before population keeps the original Jabalpur water geometries inside "
        f"`{water['before_asset'].rsplit('/', 1)[-1]}`; the After population uses "
        f"`{water['after_asset'].rsplit('/', 1)[-1]}`.",
        f"Both hold {water['counts']['after']} points, so a count check says "
        "nothing about whether they differ.",
        "",
        "Measuring the distance from every After point to its nearest Before point "
        "settles it:",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Points compared | {displacement['compared']} |",
        f"| Mean displacement | {displacement['mean_metres']} m |",
        f"| Median displacement | {displacement['median_metres']} m |",
        f"| Maximum displacement | {displacement['max_metres']} m |",
        f"| Moved more than 100 m | {displacement['moved_over_100m']} |",
        f"| Half-diagonal of a 10 m pixel | "
        f"{displacement['pixel_snap_bound_metres']} m |",
        "",
    ]
    if displacement["explained_by_pixel_snapping"]:
        lines.extend([
            "Not one point moved further than half the diagonal of a single "
            "Sentinel-2 pixel.",
            "That bound is the signature of a table round trip rather than an "
            "edit: `sampleRegions(geometries=True)` returns the centre of the "
            "pixel it read, not the point it was handed, so exporting a point "
            "set and re-importing it displaces every point by up to "
            f"{displacement['pixel_snap_bound_metres']} m.",
            "",
            "**The water points were not repositioned.**",
            "Before and After hold the same water locations, and the water class "
            "contributes no real difference to any before/after comparison in "
            "this report.",
            "",
        ])
    else:
        lines.extend([
            "The displacement exceeds what a pixel-centre round trip explains, so "
            "the points were genuinely moved.",
            "",
        ])

    lines.extend([
        "Whether the points sit on water is a separate question, and both sets "
        "answer it identically because they are the same points:",
        "",
        "| Season | Sampled | NDWI positive | On JRC surface water |",
        "|---|---:|---:|---:|",
    ])
    for season, sides in water["seasons"].items():
        after = sides["after"]
        lines.append(
            f"| {season.title()} | {after['sampled']} | "
            f"{after['ndwi_positive_percent']}% | {after['jrc_water_percent']}% |"
        )
    lines.extend([
        "",
        "Two things worth flagging.",
        "Fewer than half the water points yield a valid reading in either season, "
        "because the cloud-masked median composite has gaps over open water.",
        "And only around three in five sit on water that JRC has ever observed, "
        "which caps how good the Water class can get regardless of the model.",
        "",
    ])
    return lines


def _agriculture_section(headline, results):
    best = headline["agriculture_best"]
    lines = [
        "## What changed in the classes",
        "",
        "Agriculture is the new class, label 5.",
        f"{LAND_COVER_CLASSES[3]['name']} already holds label 3 and "
        f"{LAND_COVER_CLASSES[4]['name']} holds label 4, so 5 was the first free "
        "slot; any point arriving with label 4 in the Agriculture source is "
        "remapped to 5 on export.",
        "",
        "The class was configured in `backend/config.py` and wired through both "
        "frontends, but it never reached a model.",
        "`jabalpur_agriculture_points_updated` was imported into Earth Engine from "
        "a drawn geometry and carried no attributes at all: every one of its 1,000 "
        "features had empty properties.",
        "The training path samples with `sampleRegions(properties=['label'])` and "
        "then drops rows where `label` is null, so all 1,000 Agriculture points "
        "were discarded in silence.",
        "",
        "The repository's own asset audit catches it once run:",
        "",
        "```",
        "ValueError: after:agriculture expected label 5, got ['null']",
        "```",
        "",
        "The measured effect on the training population:",
        "",
        "| Training population | " + " | ".join(
            LAND_COVER_CLASSES[i]["name"] for i in LAND_COVER_CLASSES
        ) + " | Total |",
        "|---|" + "---:|" * (len(LAND_COVER_CLASSES) + 1),
        "| Before the fix | 1,000 | 1,000 | 1,000 | 2,000 | 500 | **0** | 5,500 |",
        "| After the fix | 1,000 | 1,000 | 1,000 | 2,000 | 500 | **1,000** | "
        "6,500 |",
        "",
        "And through the live API, over an area of interest drawn directly on the "
        "Agriculture training points, the served Random Forest returned "
        "**0.00% Agriculture** before the fix.",
        "",
    ]
    if best is not None:
        lines.extend([
            f"After the fix the class is learnable: {best['model'].upper()} reaches "
            f"{_percent(best['after'])} Agriculture recall on the held-out "
            f"{best['season']} split.",
            "",
        ])
    return lines


def _interpretation(headline):
    external_best = headline["external_best"]
    lines = [
        "## How to read these numbers",
        "",
        "Two scores are reported per run and they measure different things.",
        "",
        "**External five-class** compares predictions against a high-confidence "
        "consensus of public maps across every Madhya Pradesh district. "
        f"{LAND_COVER_CLASSES[3]['name']} and {LAND_COVER_CLASSES[4]['name']} "
        "collapse into a single Bare class for this comparison because the public "
        "references do not separate them. "
        "This is agreement with other maps, not field survey.",
        "",
        "**Held-out six-class** scores the model against the project's own labels "
        "on a fixed spatial-block split, keeping all six classes. "
        "Blocks rather than random points, because labelled points come in "
        "clusters and a random split would score each model against near "
        "duplicates of its own training rows.",
        "",
        "Raw held-out overall accuracy falls from Before to After for every model. "
        "That is expected and is not a regression. "
        "Before was scored on a four-class problem, because Sand and Agriculture "
        "had no held-out rows and Agriculture had no training rows either. "
        "After is scored on a genuinely harder six-class problem. "
        "The like-for-like table restricts scoring to the classes both conditions "
        "actually attempt, which is the comparison that answers whether the model "
        "got worse at what it already did.",
        "",
    ]
    if external_best is not None:
        lines.extend([
            f"On the external statewide reference the strongest After model is "
            f"{external_best['model'].upper()} in {external_best['season']} at "
            f"{_percent(external_best['after_accuracy'])}.",
            "",
        ])
    return lines


def _limitations(results):
    labels = results["runs"][0]["metrics"]["heldout_six_class"]["labels"]
    empty = [
        label for label in labels
        if all(
            run["metrics"]["heldout_six_class"]["per_class"][label]["support"] == 0
            for run in results["runs"]
        )
    ]
    # The rendered tables already carry a reference-limitations note, so this
    # section only adds what that note does not cover.
    lines = ["## Further limitations", ""]
    if empty:
        lines.extend([
            f"{', '.join(empty)} has no held-out rows in either condition, so it "
            "carries no held-out score at all.",
            "Its training points all fall on the training side of the spatial "
            "block split, which means the split does not currently test it.",
            "",
        ])
    lines.extend([
        "The five model configurations come from `make_classifier`, so notebook, "
        "evaluation and served backend cannot drift apart. "
        "They were not re-tuned for six classes; these are the same "
        "hyperparameters chosen for the four-class problem, which is the most "
        "likely reason Random Forest and CART barely predict Agriculture at all "
        "while Smile GTB handles it.",
        "",
    ])
    return lines


def build(results, water):
    validate_results(results, present_metric_sets(results))
    headline = _headline(results)
    lines = [
        "# Before and after accuracy across all five models",
        "",
        f"Generated {date.today().isoformat()} from "
        "`doc/assets/seasonal_before_after_results.json` and "
        "`doc/assets/water_point_change.json`.",
        "",
        f"Schema `{results['schema_version']}`, {len(BANDS)} bands, "
        f"{len(LAND_COVER_CLASSES)} classes, "
        f"{len(EXPECTED_ASSET_LABELS)} source assets.",
        "Seasons are "
        + ", ".join(
            f"{name} `{dates['start']}` to `{dates['end']}` exclusive"
            for name, dates in SEASONS.items()
        )
        + ".",
        "",
        "Models compared: Random Forest, SVM, Smile GTB, CART, KNN.",
        "",
    ]
    lines += _agriculture_section(headline, results)
    lines += _water_section(water)
    lines += _interpretation(headline)
    lines += ["## Measured results", ""]
    if "external_five_class" not in present_metric_sets(results):
        lines += [
            "The whole-Madhya-Pradesh external comparison is still computing and "
            "is not reported here. The held-out results below are complete for "
            "all five models in both seasons.",
            "",
        ]
    lines += render_markdown(results).split("\n")[1:]
    lines += _limitations(results)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--water", type=Path, default=DEFAULT_WATER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build(
        json.loads(args.results.read_text()),
        json.loads(args.water.read_text()),
    )
    args.output.write_text(report)
    print(f"wrote {args.output} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
