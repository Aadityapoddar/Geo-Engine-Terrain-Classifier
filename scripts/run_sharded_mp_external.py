#!/usr/bin/env python3
"""Checkpoint whole-MP external confusion matrices one district at a time."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import BANDS, SEASONS  # noqa: E402
from backend.gee_classifier import (  # noqa: E402
    build_sentinel_composite,
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    exclude_training_neighbours,
    madhya_pradesh_districts,
)
from evaluation.runner import MODEL_NAMES  # noqa: E402
from scripts.run_direct_full_asset_evaluation import _populations  # noqa: E402
from scripts.run_seasonal_evaluation import (  # noqa: E402
    BLOCK_SEED,
    _collapsed_external_predictions,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _call_with_retry(label, function, retries):
    error = None
    for attempt in range(1, retries + 1):
        try:
            return function()
        except ee.EEException as caught:
            error = caught
            print(f"retry {label} attempt={attempt}: {caught}", flush=True)
            time.sleep(5 * attempt)
    raise error


def _district_external(district, name, season, leakage):
    dates = SEASONS[season]
    region = district.geometry()
    reference = build_reference_image(
        region, season, dates["start"], dates["end"]
    )
    samples = reference.stratifiedSample(
        numPoints=0,
        classBand="reference",
        region=region,
        scale=10,
        classValues=list(REFERENCE_LABELS),
        classPoints=[20] * len(REFERENCE_LABELS),
        seed=BLOCK_SEED,
        geometries=True,
        tileScale=4,
    ).map(lambda feature: feature.set({"district": name, "season": season}))
    samples = exclude_training_neighbours(samples, leakage)
    composite = build_sentinel_composite(
        region, start_date=dates["start"], end_date=dates["end"]
    )
    return (
        composite.select(BANDS)
        .sampleRegions(
            collection=samples,
            properties=["reference", "district", "season"],
            scale=10,
            tileScale=8,
            geometries=True,
        )
        .filter(ee.Filter.notNull(BANDS + ["reference", "district"]))
    )


def _classifiers(populations, season):
    dates = SEASONS[season]
    output = {}
    for condition in ("before", "after"):
        full = sample_training_points(
            populations[condition],
            start_date=dates["start"],
            end_date=dates["end"],
        )
        for model in MODEL_NAMES:
            output[f"{condition}-{model}"] = make_classifier(model).train(
                features=full,
                classProperty="label",
                inputProperties=BANDS,
            )
    return output


def _evaluate_district(external, classifiers, keys=None):
    keys = list(classifiers) if keys is None else list(keys)
    payload = {"sample_count": external.size()}
    for key in keys:
        classifier = classifiers[key]
        classified = _collapsed_external_predictions(external.classify(classifier))
        payload[key] = classified.errorMatrix(
            "reference", "external_prediction", list(REFERENCE_LABELS)
        ).array().toList()
    return ee.Dictionary(payload).getInfo()


def _evaluate_resilient(external, classifiers, keys, retries, shard_id):
    """Split classifier batches when a district exceeds interactive limits."""
    error = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"evaluate {shard_id} classifiers={len(keys)} attempt={attempt}",
                flush=True,
            )
            return _evaluate_district(external, classifiers, keys)
        except ee.EEException as caught:
            error = caught
            print(f"retry {shard_id}: {caught}", flush=True)
            time.sleep(5 * attempt)
    if len(keys) == 1:
        raise error
    midpoint = len(keys) // 2
    print(f"split {shard_id} classifiers={len(keys)}", flush=True)
    left = _evaluate_resilient(
        external, classifiers, keys[:midpoint], retries, shard_id
    )
    right = _evaluate_resilient(
        external, classifiers, keys[midpoint:], retries, shard_id
    )
    if left["sample_count"] != right["sample_count"]:
        raise ValueError(f"Sample count changed while splitting {shard_id}")
    return {
        "sample_count": left["sample_count"],
        **{key: value for key, value in left.items() if key != "sample_count"},
        **{key: value for key, value in right.items() if key != "sample_count"},
    }


def run(args):
    _call_with_retry("initialize", init_ee, args.retries)
    populations = _populations(args.agriculture_geojson.resolve())
    leakage = populations["before"].merge(populations["after"])
    districts = madhya_pradesh_districts()
    names = _call_with_retry(
        "district-list",
        lambda: districts.aggregate_array("ADM2_NAME").sort().getInfo(),
        args.retries,
    )
    assigned = [
        name for index, name in enumerate(names)
        if index % args.worker_count == args.worker_index
    ]
    output = args.output or (
        REPO / "doc" / "assets" /
        f"mp_external_shards_worker_{args.worker_index}.json"
    )
    state = json.loads(output.read_text()) if output.exists() else {
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "district_count_total": len(names),
        "assigned_districts": assigned,
        "shards": {},
    }
    for season in ("winter", "summer"):
        classifiers = _call_with_retry(
            f"{season}-classifiers",
            lambda: _classifiers(populations, season),
            args.retries,
        )
        for name in assigned:
            shard_id = f"{season}:{name}"
            if shard_id in state["shards"]:
                print(f"skip {shard_id}", flush=True)
                continue
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            external = _district_external(district, name, season, leakage)
            payload = _evaluate_resilient(
                external,
                classifiers,
                list(classifiers),
                args.retries,
                shard_id,
            )
            state["shards"][shard_id] = {
                "season": season,
                "district": name,
                "sample_count": payload.pop("sample_count"),
                "matrices": payload,
            }
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(output, state)
            print(
                f"landed {shard_id} samples="
                f"{state['shards'][shard_id]['sample_count']}",
                flush=True,
            )
    print(f"wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agriculture-geojson", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    run(args)


if __name__ == "__main__":
    main()
