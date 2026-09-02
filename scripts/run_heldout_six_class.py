#!/usr/bin/env python3
"""Compute the held-out six-class runs for one or both training conditions.

The whole-MP external five-class scores come from the district-sharded runner;
this fills in the other half of the report. Runs are checkpointed by run_id, so
the Before runs already in the file are skipped rather than recomputed.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import BANDS, LAND_COVER_CLASSES, SEASONS  # noqa: E402
from backend.gee_classifier import init_ee, make_classifier  # noqa: E402
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.runner import MODEL_NAMES, SEASON_NAMES  # noqa: E402
from scripts.run_direct_full_asset_evaluation import (  # noqa: E402
    _populations,
    _seasonal_tables,
)
from scripts.run_seasonal_evaluation import _matrix_payload  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "direct_before_heldout_checkpoint.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(output, conditions):
    init_ee()
    populations = _populations()
    state = (
        json.loads(output.read_text()) if output.exists()
        else {"status": "in-progress", "runs": []}
    )
    completed = {run["run_id"] for run in state["runs"]}
    class_labels = [LAND_COVER_CLASSES[index]["name"] for index in LAND_COVER_CLASSES]

    for season in SEASON_NAMES:
        for condition in conditions:
            pending = [
                model for model in MODEL_NAMES
                if f"{season}-{condition}-{model}" not in completed
            ]
            if not pending:
                print(f"skip {season}-{condition}: complete", flush=True)
                continue
            full, split_train, heldout = _seasonal_tables(
                populations[condition], season)
            full_count = full.size().getInfo()
            split_count = split_train.size().getInfo()
            heldout_count = heldout.size().getInfo()
            print(
                f"{season}-{condition}: full={full_count} "
                f"split_train={split_count} heldout={heldout_count}",
                flush=True,
            )
            if heldout_count == 0:
                raise ValueError(f"{season}-{condition} has an empty held-out split")
            for model in pending:
                run_id = f"{season}-{condition}-{model}"
                print(f"evaluate {run_id}", flush=True)
                classifier = make_classifier(model).train(
                    features=split_train,
                    classProperty="label",
                    inputProperties=BANDS,
                )
                payload = _matrix_payload(
                    heldout.classify(classifier),
                    "label",
                    "classification",
                    list(LAND_COVER_CLASSES),
                )
                metrics = metrics_from_matrix(payload["matrix"], class_labels)
                state["runs"].append({
                    "run_id": run_id,
                    "season": season,
                    "condition": condition,
                    "model": model,
                    "training_count": full_count,
                    "heldout_training_count": split_count,
                    "metrics": {"heldout_six_class": metrics},
                })
                state["generated_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(output, state)
                print(
                    f"landed {run_id}: heldout="
                    f"{metrics['overall_accuracy']:.6f}",
                    flush=True,
                )

    expected = {
        f"{season}-{condition}-{model}"
        for season in SEASON_NAMES
        for condition in ("before", "after")
        for model in MODEL_NAMES
    }
    if {run["run_id"] for run in state["runs"]} == expected:
        state["status"] = "complete"
    _write_json(output, state)
    print(f"wrote {output} ({len(state['runs'])}/20 runs)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conditions", nargs="+", default=["before", "after"],
        choices=["before", "after"])
    args = parser.parse_args()
    run(args.output.resolve(), args.conditions)


if __name__ == "__main__":
    main()
