#!/usr/bin/env python3
"""Validate and aggregate district-sharded whole-MP confusion matrices."""

import argparse
import json
from pathlib import Path

from evaluation.metrics import metrics_from_matrix
from evaluation.references import REFERENCE_LABELS
from evaluation.runner import MODEL_NAMES


def _empty_matrix():
    return [[0 for _ in REFERENCE_LABELS] for _ in REFERENCE_LABELS]


def _add_matrix(total, value):
    for row in range(len(total)):
        for column in range(len(total[row])):
            total[row][column] += value[row][column]


def aggregate(paths):
    shards = {}
    expected_districts = set()
    expected_total = None
    for path in paths:
        payload = json.loads(path.read_text())
        expected_districts.update(payload["assigned_districts"])
        if expected_total is None:
            expected_total = payload["district_count_total"]
        elif expected_total != payload["district_count_total"]:
            raise ValueError("Worker district totals disagree")
        overlap = set(shards).intersection(payload["shards"])
        if overlap:
            raise ValueError(f"Duplicate shards: {sorted(overlap)}")
        shards.update(payload["shards"])
    if len(expected_districts) != expected_total:
        raise ValueError(
            f"Worker assignments cover {len(expected_districts)}/{expected_total} districts"
        )

    output = {
        "district_count": expected_total,
        "shard_count": len(shards),
        "runs": [],
    }
    required = {
        f"{season}:{district}"
        for season in ("winter", "summer")
        for district in expected_districts
    }
    missing = required - set(shards)
    if missing:
        raise ValueError(f"Missing {len(missing)} shards; first={sorted(missing)[:5]}")

    labels = list(REFERENCE_LABELS.values())
    for season in ("winter", "summer"):
        season_shards = [
            shard for shard in shards.values() if shard["season"] == season
        ]
        sample_count = sum(shard["sample_count"] for shard in season_shards)
        district_coverage = sum(shard["sample_count"] > 0 for shard in season_shards)
        for condition in ("before", "after"):
            for model in MODEL_NAMES:
                matrix = _empty_matrix()
                key = f"{condition}-{model}"
                for shard in season_shards:
                    _add_matrix(matrix, shard["matrices"][key])
                metrics = metrics_from_matrix(matrix, labels)
                if metrics["sample_count"] != sample_count:
                    raise ValueError(f"Sample-count mismatch for {season}-{key}")
                metrics["district_coverage"] = district_coverage
                output["runs"].append({
                    "run_id": f"{season}-{condition}-{model}",
                    "season": season,
                    "condition": condition,
                    "model": model,
                    "metrics": {"external_five_class": metrics},
                })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "shards": result["shard_count"],
        "runs": len(result["runs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
