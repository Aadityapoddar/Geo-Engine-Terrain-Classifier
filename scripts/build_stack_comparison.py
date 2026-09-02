#!/usr/bin/env python3
"""Combine the development and test runs of the candidate band stacks.

scripts/district_experiments.py writes one file per district set. This joins
them into the single artefact the paper and the figures read, and attaches the
district-block confidence intervals and paired tests that make the comparison
answerable: whether one stack beats another is a claim about a difference, and
a difference whose interval straddles zero is not a reduction that "improved
accuracy".

    python scripts/build_stack_comparison.py \\
        --development doc/assets/band_stacks_development_v4.json \\
        --test doc/assets/band_stacks_test_v4.json \\
        --selection doc/assets/band_selection_v4.json \\
        --output doc/assets/band_stack_comparison_v4.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import REFERENCE_LABELS  # noqa: E402
from scripts.merge_experiment_shards import merge  # noqa: E402
from scripts.spatial_uncertainty import (  # noqa: E402
    BOOTSTRAP_SEED,
    block_bootstrap,
    paired_comparison,
)

LABELS = list(REFERENCE_LABELS.values())


def per_district(payload, season):
    """{district: {stack: matrix}} for one season."""
    out = {}
    for shard in payload["shards"].values():
        if shard["season"] != season:
            continue
        out[shard["district"]] = {
            key.split(":", 1)[1]: np.array(matrix, dtype=float)
            for key, matrix in shard["matrices"].items()
        }
    return out


def summarise(districts, stacks, replicates, rng):
    pooled = {}
    for stack in stacks:
        matrix = sum(districts[name][stack] for name in districts)
        metrics = metrics_from_matrix(matrix.astype(int).tolist(), LABELS)
        block = block_bootstrap(
            {name: districts[name][stack] for name in districts},
            replicates, rng)
        pooled[stack] = {
            "overall_accuracy": metrics["overall_accuracy"],
            "overall_accuracy_ci95": block["overall_accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "sample_count": metrics["sample_count"],
            "district_count": len(districts),
            "per_class_recall": {
                label: metrics["per_class"][label]["recall"] for label in LABELS},
            "per_class_precision": {
                label: metrics["per_class"][label]["precision"]
                for label in LABELS},
        }
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", nargs="+", type=Path, required=True)
    parser.add_argument("--test", nargs="+", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()

    # Accepts the worker files directly, so a sharded run needs no separate
    # merge before the comparison can be built.
    development, _ = merge(args.development)
    test, _ = merge(args.test)
    selection = json.loads(args.selection.read_text())
    stacks = selection["stacks"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Widest first, so the reference point leads and the table reads as a
    # reduction rather than as an arbitrary ordering.
    order = sorted(stacks, key=lambda name: (-len(stacks[name]), name))

    output = {
        "training_schema_version": development.get("training_schema_version"),
        "stacks": stacks,
        "stacks_order": order,
        "droppable_by_season": selection.get("droppable_by_season"),
        "droppable_in_both": selection.get("droppable_in_both"),
        "protocol": (
            "Every stack is chosen on the development districts. The test "
            "districts are scored once, afterwards, with whatever that chose."
        ),
        "seasons": {},
    }

    for season in ("winter", "summer"):
        dev_districts = per_district(development, season)
        test_districts = per_district(test, season)
        if not dev_districts or not test_districts:
            continue
        present = [name for name in order
                   if name in next(iter(dev_districts.values()))]
        payload = {
            "development": summarise(dev_districts, present, args.replicates, rng),
            "test": summarise(test_districts, present, args.replicates, rng),
            "test_paired": [
                paired_comparison(test_districts, left, right,
                                  args.replicates, rng)
                for left, right in combinations(present, 2)
            ],
        }
        payload["selected_on_development"] = max(
            present,
            key=lambda name: payload["development"][name]["overall_accuracy"])
        output["seasons"][season] = payload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    for season, payload in output["seasons"].items():
        print(f"\n== {season} ==")
        for name in output["stacks_order"]:
            if name not in payload["development"]:
                continue
            dev = payload["development"][name]
            tst = payload["test"][name]
            low, high = tst["overall_accuracy_ci95"]
            print(f"  {name:14s} {len(stacks[name]):2d} bands  "
                  f"dev {dev['overall_accuracy'] * 100:5.2f}%  "
                  f"test {tst['overall_accuracy'] * 100:5.2f}% "
                  f"[{low * 100:5.2f}, {high * 100:5.2f}]  "
                  f"open-land recall "
                  f"{(tst['per_class_recall'].get('Open Land') or 0) * 100:5.1f}%")
        chosen = payload["selected_on_development"]
        print(f"  selected on development: {chosen} "
              f"({len(stacks[chosen])} bands) -> test "
              f"{payload['test'][chosen]['overall_accuracy'] * 100:.2f}%")
        for comparison in payload["test_paired"]:
            if chosen not in (comparison["left"], comparison["right"]):
                continue
            sign = 1 if comparison["left"] == chosen else -1
            other = comparison["right"] if sign == 1 else comparison["left"]
            low, high = sorted(v * sign for v in comparison["ci95"])
            print(f"    {chosen} - {other}: "
                  f"{comparison['mean_district_oa_difference'] * sign * 100:+.2f} pp "
                  f"[{low * 100:+.2f}, {high * 100:+.2f}] "
                  f"p={comparison['wilcoxon_p']:.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
