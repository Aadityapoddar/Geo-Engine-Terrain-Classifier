#!/usr/bin/env python3
"""Run the 20 seasonal evaluations without batch-exported sample tables.

This is the restricted-quota fallback. External MP scores train on every
seasonally valid point in the selected Before or After population. The separate
heldout score retains the fixed spatial split and therefore uses a second
classifier trained only on that split's training partition.
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

from backend.config import (  # noqa: E402
    BEFORE_TRAINING_TABLE,
    BANDS,
    FEATURE_COLLECTIONS,
    LAND_COVER_CLASSES,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import (  # noqa: E402
    build_sentinel_composite,
    init_ee,
    make_classifier,
    merge_feature_collections,
    sample_training_points,
)
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    madhya_pradesh_districts,
    sample_mp_reference,
)
from evaluation.runner import MODEL_NAMES  # noqa: E402
from scripts.run_seasonal_evaluation import (  # noqa: E402
    BLOCK_SEED,
    _assign_block,
    _collapsed_external_predictions,
    _matrix_payload,
)


DEFAULT_OUTPUT = REPO / "doc" / "assets" / "seasonal_before_after_results.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _inline_agriculture(path):
    payload = json.loads(path.read_text())
    features = payload.get("features", [])
    if len(features) != 1000:
        raise ValueError(f"Expected 1000 Agriculture features, got {len(features)}")
    if {feature.get("geometry", {}).get("type") for feature in features} != {"Point"}:
        raise ValueError("Agriculture fallback must contain only Point geometries")
    if {feature.get("properties", {}).get("label") for feature in features} != {5}:
        raise ValueError("Agriculture fallback must contain only label 5")
    return ee.FeatureCollection(payload)


def _populations(agriculture_path):
    before = ee.FeatureCollection(BEFORE_TRAINING_TABLE)
    current_paths = [
        path for name, path in FEATURE_COLLECTIONS.items() if name != "agriculture"
    ]
    after = merge_feature_collections(current_paths).merge(
        _inline_agriculture(agriculture_path)
    )
    return {"before": before, "after": after}


def _seasonal_tables(points, season):
    dates = SEASONS[season]
    full = sample_training_points(
        points, start_date=dates["start"], end_date=dates["end"]
    )
    blocked = points.map(_assign_block)
    split_train = sample_training_points(
        blocked.filter(ee.Filter.lt("block", 7)),
        start_date=dates["start"],
        end_date=dates["end"],
    )
    heldout = sample_training_points(
        blocked.filter(ee.Filter.gte("block", 7)),
        start_date=dates["start"],
        end_date=dates["end"],
    )
    return full, split_train, heldout


def _external_table(season, leakage_sources):
    dates = SEASONS[season]
    districts = madhya_pradesh_districts()
    region = districts.geometry()
    reference = build_reference_image(
        region, season, dates["start"], dates["end"]
    )
    candidates = sample_mp_reference(reference, season, leakage_sources)
    composite = build_sentinel_composite(
        region, start_date=dates["start"], end_date=dates["end"]
    )
    return (
        composite.select(BANDS)
        .sampleRegions(
            collection=candidates,
            properties=["reference", "district", "season"],
            scale=10,
            tileScale=8,
            geometries=True,
        )
        .filter(ee.Filter.notNull(BANDS + ["reference", "district"]))
    )


def _new_results(agriculture_path):
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "bands": BANDS,
        "seasons": SEASONS,
        "reference": {
            "external_labels": REFERENCE_LABELS,
            "soil_sand_collapsed": True,
            "sampling_seed": BLOCK_SEED,
            "training_exclusion_metres": 100,
            "coverage": "all Madhya Pradesh districts",
        },
        "training_protocol": {
            "external_five_class": "all seasonally valid population points",
            "heldout_six_class": "fixed 70/30 spatial block split",
        },
        "agriculture_fallback": {
            "path": str(agriculture_path),
            "source": "Earth Engine Code Editor agriculture_points import",
            "feature_count": 1000,
        },
        "runs": [],
    }


def run(agriculture_path, output):
    init_ee()
    populations = _populations(agriculture_path)
    leakage_sources = populations["before"].merge(populations["after"])
    results = (
        json.loads(output.read_text()) if output.exists()
        else _new_results(agriculture_path)
    )
    completed = {item["run_id"] for item in results["runs"]}
    project_labels = [
        LAND_COVER_CLASSES[index]["name"] for index in LAND_COVER_CLASSES
    ]

    for season in ("winter", "summer"):
        print(f"build {season} external whole-MP table", flush=True)
        external = _external_table(season, leakage_sources)
        external_count = external.size().getInfo()
        district_count = external.aggregate_count_distinct("district").getInfo()
        print(
            f"external {season}: {external_count} samples, "
            f"{district_count} districts",
            flush=True,
        )
        for condition in ("before", "after"):
            full, split_train, heldout = _seasonal_tables(
                populations[condition], season
            )
            full_count = full.size().getInfo()
            split_count = split_train.size().getInfo()
            heldout_count = heldout.size().getInfo()
            print(
                f"{season}-{condition}: full={full_count}, "
                f"split_train={split_count}, heldout={heldout_count}",
                flush=True,
            )
            for model in MODEL_NAMES:
                run_id = f"{season}-{condition}-{model}"
                if run_id in completed:
                    print(f"skip {run_id}", flush=True)
                    continue
                print(f"evaluate {run_id}", flush=True)
                full_classifier = make_classifier(model).train(
                    features=full,
                    classProperty="label",
                    inputProperties=BANDS,
                )
                split_classifier = make_classifier(model).train(
                    features=split_train,
                    classProperty="label",
                    inputProperties=BANDS,
                )
                external_payload = _matrix_payload(
                    _collapsed_external_predictions(
                        external.classify(full_classifier)
                    ),
                    "reference",
                    "external_prediction",
                    list(REFERENCE_LABELS),
                    district_field="district",
                )
                heldout_payload = _matrix_payload(
                    heldout.classify(split_classifier),
                    "label",
                    "classification",
                    list(LAND_COVER_CLASSES),
                )
                external_metrics = metrics_from_matrix(
                    external_payload["matrix"], list(REFERENCE_LABELS.values())
                )
                external_metrics["district_coverage"] = external_payload[
                    "district_coverage"
                ]
                heldout_metrics = metrics_from_matrix(
                    heldout_payload["matrix"], project_labels
                )
                results["runs"].append({
                    "run_id": run_id,
                    "season": season,
                    "condition": condition,
                    "model": model,
                    "training_count": full_count,
                    "heldout_training_count": split_count,
                    "metrics": {
                        "external_five_class": external_metrics,
                        "heldout_six_class": heldout_metrics,
                    },
                })
                results["updated_at"] = _now()
                _write_json(output, results)
                print(
                    f"landed {run_id}: external="
                    f"{external_metrics['overall_accuracy']:.6f}, heldout="
                    f"{heldout_metrics['overall_accuracy']:.6f}",
                    flush=True,
                )
    print(f"wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agriculture-geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.agriculture_geojson.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
