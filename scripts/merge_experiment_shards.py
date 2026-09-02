#!/usr/bin/env python3
"""Merge the worker files from a sharded district experiment, then summarise.

scripts/district_experiments.py stripes its districts across workers, so each
one holds a slice of the evidence. Summarising a slice would report a third of
the districts as if it were all of them, so the workers do not summarise; this
does, once the slices are back together.

Refuses to merge across schema versions or across modes, for the same reason
the shard aggregator does: two files that describe different pipelines cannot
be summed, and finding that out from a suspicious number later is much worse
than finding out here.

    python scripts/merge_experiment_shards.py doc/assets/band_stacks_dev_v4_w*.json \\
        --output doc/assets/band_stacks_development_v4.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.district_experiments import summarise  # noqa: E402


def merge(paths):
    merged = None
    for path in sorted(paths):
        payload = json.loads(Path(path).read_text())
        if merged is None:
            merged = {key: value for key, value in payload.items()
                      if key not in ("shards", "assigned_districts",
                                     "worker_index", "worker_count",
                                     "scores", "best_per_model", "ablation",
                                     "pooled_matrices")}
            merged["shards"] = {}
            merged["worker_files"] = []
        for field in ("training_schema_version", "mode", "districts_used"):
            if payload.get(field) != merged.get(field):
                raise SystemExit(
                    f"{path} has {field}={payload.get(field)!r}, expected "
                    f"{merged.get(field)!r}; these are different experiments")
        overlap = set(merged["shards"]) & set(payload["shards"])
        if overlap:
            raise SystemExit(f"{path} repeats shards: {sorted(overlap)[:5]}")
        merged["shards"].update(payload["shards"])
        merged["worker_files"].append(Path(path).name)
    if merged is None:
        raise SystemExit("no input files")

    expected = set(merged.get("districts") or [])
    seasons = {shard["season"] for shard in merged["shards"].values()}
    for season in sorted(seasons):
        present = {shard["district"] for shard in merged["shards"].values()
                   if shard["season"] == season}
        missing = expected - present
        if missing:
            print(f"  warning: {season} is missing {len(missing)} districts: "
                  f"{', '.join(sorted(missing)[:6])}")
    return merged, sorted(seasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged, seasons = merge(args.paths)
    for season in seasons:
        summarise(merged, season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nwrote {args.output}: {len(merged['shards'])} shards from "
          f"{len(merged['worker_files'])} workers")


if __name__ == "__main__":
    main()
