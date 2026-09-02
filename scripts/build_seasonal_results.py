#!/usr/bin/env python3
"""Merge the two halves of the seasonal evidence into one validated results file.

The whole-MP external five-class metrics come from the district-sharded runner
(aggregated across all 48 districts); the held-out six-class metrics come from
the fixed spatial-block split. Both cover the same 20 runs, and the report
contract requires each run to carry both. This joins them on run_id and refuses
to write anything the report validator would not accept.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import BANDS, SEASONS, TRAINING_SCHEMA_VERSION  # noqa: E402
from evaluation.report import (  # noqa: E402
    present_metric_sets,
    validate_results,
)
from evaluation.runner import iter_run_specs  # noqa: E402

ASSETS = REPO / "doc" / "assets"
DEFAULT_EXTERNAL = ASSETS / "mp_external_aggregate.json"
DEFAULT_HELDOUT = ASSETS / "direct_before_heldout_checkpoint.json"
DEFAULT_OUTPUT = ASSETS / "seasonal_before_after_results.json"


def _index(payload, metric_set, source):
    indexed = {}
    for run in payload["runs"]:
        metrics = run["metrics"].get(metric_set)
        if metrics is None:
            raise ValueError(f"{source}: {run['run_id']} has no {metric_set}")
        indexed[run["run_id"]] = (run, metrics)
    return indexed


def merge(external_payload, heldout_payload):
    """Join the two halves on run_id. The external half may be absent.

    Its district-sharded runner takes hours and is the one that keeps losing its
    network, so the report is built from whatever is complete rather than held
    hostage to it. Partial external coverage is not accepted: it is all 20 runs
    or none.
    """
    if external_payload:
        recorded = external_payload.get("training_schema_version") or "an unstamped schema"
        if recorded != TRAINING_SCHEMA_VERSION:
            raise ValueError(
                f"External half was computed under {recorded} but the current "
                f"schema is {TRAINING_SCHEMA_VERSION}. Class 3 differs between "
                "them, so merging would put two class inventories in one report."
            )
    external = (
        _index(external_payload, "external_five_class", "external")
        if external_payload else {}
    )
    heldout = _index(heldout_payload, "heldout_six_class", "heldout")

    runs = []
    for spec in iter_run_specs():
        if spec.run_id not in heldout:
            raise ValueError(f"heldout source is missing {spec.run_id}")
        if external and spec.run_id not in external:
            raise ValueError(f"external source is missing {spec.run_id}")
        heldout_run, heldout_metrics = heldout[spec.run_id]
        metrics = {"heldout_six_class": heldout_metrics}
        entry = {
            "run_id": spec.run_id,
            "season": spec.season,
            "condition": spec.condition,
            "model": spec.model,
            "training_count": heldout_run.get("training_count"),
            "heldout_training_count": heldout_run.get("heldout_training_count"),
            "metrics": metrics,
        }
        if external:
            _, external_metrics = external[spec.run_id]
            metrics["external_five_class"] = external_metrics
            entry["external_sample_count"] = external_metrics["sample_count"]
        runs.append(entry)

    external_payload = external_payload or {}
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "bands": BANDS,
        "seasons": SEASONS,
        "sources": {
            "external_five_class": {
                "method": "district-sharded whole-MP public-map consensus",
                "district_count": external_payload.get("district_count"),
                "shard_count": external_payload.get("shard_count"),
            },
            "heldout_six_class": {
                "method": "fixed 70/30 spatial block split on project labels",
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    external = (
        json.loads(args.external.read_text()) if args.external.exists() else None
    )
    if external is None:
        print(f"note: {args.external} absent; external five-class half omitted")
    results = merge(external, json.loads(args.heldout.read_text()))
    present = present_metric_sets(results)
    validate_results(results, present)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output} ({len(results['runs'])} runs, "
          f"metric sets: {', '.join(present)})")


if __name__ == "__main__":
    main()
