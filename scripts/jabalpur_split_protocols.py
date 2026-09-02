#!/usr/bin/env python3
"""The random and spatially blocked splits inside the labelled area, under v4.

Table XI sets three protocols against each other: a random held-out split, a
spatially blocked split, and the external district consensus. Only the third
was re-measured when the pipeline changed, which would have left one table
quoting two different composites -- exactly the internal inconsistency this
revision exists to remove. This recomputes the first two on the v4 composite.

The blocked split reuses the block assignment in
scripts/run_seasonal_evaluation.py rather than inventing another one: a coarse
0.1-degree lattice, scrambled by a fixed integer hash so blocks alternate
across the district instead of splitting it into two halves that differ in
landscape as well as in membership. Blocks 0-6 train, 7-9 are held out.

The random split is the thing being argued against, so it is drawn the naive
way on purpose: a uniform 70/30 over the same points, with no spatial
constraint at all. The gap between the two numbers is the whole point.
"""

import argparse
import json
import sys
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    BAND_STACKS,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import (  # noqa: E402
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.assets import training_points  # noqa: E402
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import REFERENCE_LABELS  # noqa: E402
from evaluation.runner import MODEL_NAMES  # noqa: E402
from scripts.run_seasonal_evaluation import BLOCK_SEED, BLOCK_SPLIT, _assign_block  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "jabalpur_split_protocols.json"
LABELS = list(REFERENCE_LABELS.values())
RANDOM_TRAIN_FRACTION = 0.7


def _metrics(matrix):
    return metrics_from_matrix([[int(v) for v in row] for row in matrix], LABELS)


def _score(train_table, test_table, bands):
    """Train every model on one table and score it on the other."""
    payload = {"sample_count": test_table.size()}
    for model in MODEL_NAMES:
        classifier = make_classifier(model).train(
            features=train_table, classProperty="label", inputProperties=bands)
        payload[model] = (
            test_table.classify(classifier)
            .errorMatrix("label", "classification", list(REFERENCE_LABELS))
            .array().toList()
        )
    return ee.Dictionary(payload).getInfo()


def run(args):
    init_ee()
    bands = BAND_STACKS["b19"]
    dates = SEASONS[args.season]
    points = training_points("after")
    sampled = sample_training_points(
        points, start_date=dates["start"], end_date=dates["end"], bands=bands)

    # Random: a uniform column, thresholded. No geometry involved, which is the
    # protocol being criticised.
    randomised = sampled.randomColumn("split", BLOCK_SEED)
    random_train = randomised.filter(ee.Filter.lt("split", RANDOM_TRAIN_FRACTION))
    random_test = randomised.filter(ee.Filter.gte("split", RANDOM_TRAIN_FRACTION))

    # Blocked: assigned on the point geometry, so neighbouring points land in
    # the same block and cannot straddle the split.
    blocked = sampled.map(_assign_block)
    blocked_train = blocked.filter(ee.Filter.lt("block", BLOCK_SPLIT))
    blocked_test = blocked.filter(ee.Filter.gte("block", BLOCK_SPLIT))

    output = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "season": args.season,
        "band_stack": "b19",
        "block_split": BLOCK_SPLIT,
        "random_train_fraction": RANDOM_TRAIN_FRACTION,
        "protocols": {},
    }
    for name, (train_table, test_table) in {
        "random": (random_train, random_test),
        "spatially_blocked": (blocked_train, blocked_test),
    }.items():
        raw = _score(train_table, test_table, bands)
        sample_count = raw.pop("sample_count")
        entry = {"sample_count": sample_count, "models": {}}
        for model, matrix in raw.items():
            metrics = _metrics(matrix)
            entry["models"][model] = {
                "overall_accuracy": metrics["overall_accuracy"],
                "kappa": metrics["kappa"],
                "macro_f1": metrics["macro_f1"],
            }
        values = [value["overall_accuracy"] for value in entry["models"].values()]
        entry["overall_accuracy_range"] = [min(values), max(values)]
        output["protocols"][name] = entry
        print(f"{name:18s} n={sample_count:5d}  "
              f"{min(values) * 100:.1f}-{max(values) * 100:.1f}%  "
              + "  ".join(f"{m} {v['overall_accuracy'] * 100:.1f}"
                          for m, v in sorted(entry["models"].items())),
              flush=True)

    existing = (json.loads(args.output.read_text())
                if args.output.exists() else {"seasons": {}})
    existing.setdefault("seasons", {})[args.season] = output
    existing["training_schema_version"] = TRAINING_SCHEMA_VERSION
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter", choices=list(SEASONS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
