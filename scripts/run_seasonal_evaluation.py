#!/usr/bin/env python3
"""Prepare, inspect, evaluate, and report the fixed 20-run seasonal study.

Examples:
  python scripts/run_seasonal_evaluation.py prepare --max-submit 2
  python scripts/run_seasonal_evaluation.py status
  python scripts/run_seasonal_evaluation.py evaluate
  python scripts/run_seasonal_evaluation.py report
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
    BANDS,
    EE_ASSET_ROOT,
    LAND_COVER_CLASSES,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import (  # noqa: E402
    build_sentinel_composite,
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.assets import training_points  # noqa: E402
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    madhya_pradesh_districts,
    sample_mp_reference,
)
from evaluation.report import render_markdown, write_confusion_matrices  # noqa: E402
from evaluation.runner import iter_run_specs  # noqa: E402


ASSETS = REPO / "doc" / "assets"
MANIFEST_PATH = ASSETS / "seasonal_evaluation_manifest.json"
RESULTS_PATH = ASSETS / "seasonal_before_after_results.json"
REPORT_PATH = REPO / "doc" / "seasonal_before_after_accuracy.md"
MATRIX_DIR = ASSETS / "seasonal_confusion_matrices"
SCHEMA_SLUG = TRAINING_SCHEMA_VERSION.replace("-", "_")
BLOCK_SPLIT = 7
BLOCK_SEED = 20260807


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def asset_ids():
    prefix = f"{EE_ASSET_ROOT}/seasonal_{SCHEMA_SLUG}"
    assets = {}
    for season in SEASONS:
        for condition in ("before", "after"):
            assets[f"{season}-{condition}-full"] = (
                f"{prefix}_{season}_{condition}_full_train"
            )
            assets[f"{season}-{condition}-train"] = (
                f"{prefix}_{season}_{condition}_blocked_train"
            )
            assets[f"{season}-{condition}-heldout"] = (
                f"{prefix}_{season}_{condition}_blocked_heldout"
            )
        assets[f"{season}-external"] = f"{prefix}_{season}_mp_external"
    return assets


def _assign_block(feature):
    coordinates = feature.geometry().coordinates()
    block_x = ee.Number(coordinates.get(0)).multiply(10).floor()
    block_y = ee.Number(coordinates.get(1)).multiply(10).floor()
    block = block_x.multiply(31).add(block_y.multiply(17)).mod(10)
    return feature.set("block", block)


def _training_and_heldout_tables(condition, season):
    dates = SEASONS[season]
    blocked = training_points(condition).map(_assign_block)
    train_points = blocked.filter(ee.Filter.lt("block", BLOCK_SPLIT))
    heldout_points = blocked.filter(ee.Filter.gte("block", BLOCK_SPLIT))
    train = sample_training_points(
        train_points, start_date=dates["start"], end_date=dates["end"]
    )
    heldout = sample_training_points(
        heldout_points, start_date=dates["start"], end_date=dates["end"]
    )
    return train, heldout


def _external_table(season):
    dates = SEASONS[season]
    districts = madhya_pradesh_districts()
    region = districts.geometry()
    reference = build_reference_image(region, season, dates["start"], dates["end"])
    # Exclude proximity to either population, not merely the model being scored.
    leakage_sources = training_points("before").merge(training_points("after"))
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


def _table_for_key(key):
    parts = key.split("-")
    season = parts[0]
    if parts[1] == "external":
        return _external_table(season)
    condition, kind = parts[1], parts[2]
    if kind == "full":
        dates = SEASONS[season]
        return sample_training_points(
            training_points(condition),
            start_date=dates["start"],
            end_date=dates["end"],
        )
    train, heldout = _training_and_heldout_tables(condition, season)
    return train if kind == "train" else heldout


def _asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def _refresh_manifest(manifest):
    for key, record in manifest.get("tasks", {}).items():
        task_id = record.get("task_id")
        if task_id:
            status = ee.data.getTaskStatus(task_id)[0]
            record.update(
                state=status.get("state", "UNKNOWN"),
                error_message=status.get("error_message"),
                updated_at=_now(),
            )
    for key, asset_id in asset_ids().items():
        manifest.setdefault("assets", {})[key] = {
            "asset_id": asset_id,
            "exists": _asset_exists(asset_id),
        }
    manifest["updated_at"] = _now()
    return manifest


def cmd_prepare(args):
    init_ee()
    manifest = _load_json(MANIFEST_PATH, {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "created_at": _now(),
        "tasks": {},
        "assets": {},
    })
    _refresh_manifest(manifest)
    active = sum(
        record.get("state") in {"READY", "RUNNING"}
        for record in manifest["tasks"].values()
    )
    capacity = max(0, args.max_submit - active)
    submitted = 0
    for key, asset_id in asset_ids().items():
        if manifest["assets"][key]["exists"]:
            continue
        previous = manifest["tasks"].get(key, {})
        if previous.get("state") in {"READY", "RUNNING", "COMPLETED"}:
            continue
        if submitted >= capacity:
            break
        print(f"building {key}", flush=True)
        task = ee.batch.Export.table.toAsset(
            collection=_table_for_key(key),
            description=f"seasonal_{SCHEMA_SLUG}_{key.replace('-', '_')}",
            assetId=asset_id,
        )
        task.start()
        manifest["tasks"][key] = {
            "task_id": task.id,
            "asset_id": asset_id,
            "state": "READY",
            "submitted_at": _now(),
        }
        submitted += 1
        print(f"submitted {key}: {task.id}", flush=True)
        _write_json(MANIFEST_PATH, manifest)
    _refresh_manifest(manifest)
    _write_json(MANIFEST_PATH, manifest)
    print(json.dumps({"submitted": submitted, "active": active + submitted,
                      "manifest": str(MANIFEST_PATH)}, indent=2))


def cmd_status(_args):
    init_ee()
    manifest = _load_json(MANIFEST_PATH, {
        "schema_version": TRAINING_SCHEMA_VERSION, "tasks": {}, "assets": {}
    })
    _refresh_manifest(manifest)
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        key: {
            "exists": manifest["assets"][key]["exists"],
            "state": manifest.get("tasks", {}).get(key, {}).get("state", "NOT_SUBMITTED"),
            "task_id": manifest.get("tasks", {}).get(key, {}).get("task_id"),
        }
        for key in asset_ids()
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(row["exists"] for row in summary.values()) else 2


def _collapsed_external_predictions(classified):
    mapping = ee.List([0, 1, 2, 3, 3, 4])
    return classified.map(lambda feature: feature.set(
        "external_prediction",
        mapping.get(ee.Number(feature.get("classification")).toInt()),
    ))


def _matrix_payload(classified, actual, predicted, order, district_field=None):
    matrix = classified.errorMatrix(actual, predicted, order)
    payload = {
        "matrix": matrix.array().toList(),
        "sample_count": classified.size(),
    }
    if district_field:
        payload["district_coverage"] = classified.aggregate_count_distinct(district_field)
    return ee.Dictionary(payload).getInfo()


def _evaluate_run(spec, assets):
    full_train = ee.FeatureCollection(
        assets[f"{spec.season}-{spec.condition}-full"]
    )
    split_train = ee.FeatureCollection(
        assets[f"{spec.season}-{spec.condition}-train"]
    )
    external = ee.FeatureCollection(assets[f"{spec.season}-external"])
    heldout = ee.FeatureCollection(assets[f"{spec.season}-{spec.condition}-heldout"])
    external_classifier = make_classifier(spec.model).train(
        features=full_train, classProperty="label", inputProperties=BANDS
    )
    heldout_classifier = make_classifier(spec.model).train(
        features=split_train, classProperty="label", inputProperties=BANDS
    )
    external_classified = _collapsed_external_predictions(
        external.classify(external_classifier)
    )
    heldout_classified = heldout.classify(heldout_classifier)
    external_payload = _matrix_payload(
        external_classified, "reference", "external_prediction",
        list(REFERENCE_LABELS), district_field="district",
    )
    heldout_payload = _matrix_payload(
        heldout_classified, "label", "classification", list(LAND_COVER_CLASSES)
    )
    external_metrics = metrics_from_matrix(
        external_payload["matrix"], list(REFERENCE_LABELS.values())
    )
    external_metrics["district_coverage"] = external_payload["district_coverage"]
    heldout_metrics = metrics_from_matrix(
        heldout_payload["matrix"],
        [LAND_COVER_CLASSES[index]["name"] for index in LAND_COVER_CLASSES],
    )
    return {
        "run_id": spec.run_id,
        "season": spec.season,
        "condition": spec.condition,
        "model": spec.model,
        "training_count": full_train.size().getInfo(),
        "heldout_training_count": split_train.size().getInfo(),
        "metrics": {
            "external_five_class": external_metrics,
            "heldout_six_class": heldout_metrics,
        },
    }


def cmd_evaluate(_args):
    init_ee()
    assets = asset_ids()
    missing = [key for key, asset_id in assets.items() if not _asset_exists(asset_id)]
    if missing:
        raise SystemExit("Required exported tables missing: " + ", ".join(missing))
    results = _load_json(RESULTS_PATH, {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "bands": BANDS,
        "seasons": SEASONS,
        "reference": {
            "external_labels": REFERENCE_LABELS,
            "soil_sand_collapsed": True,
            "sampling_seed": BLOCK_SEED,
            "training_exclusion_metres": 100,
        },
        "runs": [],
    })
    completed = {run["run_id"] for run in results["runs"]}
    for spec in iter_run_specs():
        if spec.run_id in completed:
            print(f"skip {spec.run_id}", flush=True)
            continue
        print(f"evaluate {spec.run_id}", flush=True)
        results["runs"].append(_evaluate_run(spec, assets))
        results["updated_at"] = _now()
        _write_json(RESULTS_PATH, results)
    print(f"wrote {RESULTS_PATH}")


def cmd_report(_args):
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}; run evaluate first")
    results = json.loads(RESULTS_PATH.read_text())
    REPORT_PATH.write_text(render_markdown(results))
    matrices = write_confusion_matrices(results, MATRIX_DIR)
    print(json.dumps({"report": str(REPORT_PATH), "matrices": len(matrices)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--max-submit", type=int, default=2)
    prepare.set_defaults(func=cmd_prepare)
    commands.add_parser("status").set_defaults(func=cmd_status)
    commands.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    commands.add_parser("report").set_defaults(func=cmd_report)
    args = parser.parse_args()
    result = args.func(args)
    raise SystemExit(result or 0)


if __name__ == "__main__":
    main()
