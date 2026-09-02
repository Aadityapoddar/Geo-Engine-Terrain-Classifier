#!/usr/bin/env python3
"""Score many trained configurations over a district set, one pass per district.

Four experiments share this runner because they share the expensive part. The
cost of a district is the composite: Sentinel-2 median, indices, GLCM texture,
Sentinel-1, then the stratified reference sample. Once that exists, scoring one
more classifier against it is nearly free, so thirty-six configurations in a
single pass cost barely more than one.

    --mode grid       hyperparameter search over all five classifiers
    --mode groups     cumulative feature families, the attribution study
    --mode ablation   leave-one-band-out over the 27-band stack
    --mode stacks     named candidate stacks, for selection and confirmation

    --districts development   where every choice is made (the default)
    --districts test          where a frozen choice is scored, once

The split is the point. Development and test districts are disjoint and were
partitioned before anything was selected (evaluation/splits.py), so a stack or a
hyperparameter chosen here can afterwards be scored on the other half and that
number is an estimate rather than a maximum over a search.

    python scripts/district_experiments.py --mode ablation --season winter
    python scripts/district_experiments.py --mode stacks --districts test \
        --stacks doc/assets/band_selection_v4.json --season winter
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee

SOCKET_TIMEOUT_SECONDS = 600
RETRYABLE_ERRORS = (ee.EEException, OSError)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    BANDS,
    EE_ASSET_ROOT,
    FULL_BANDS,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import (  # noqa: E402
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import REFERENCE_LABELS  # noqa: E402
from evaluation.splits import district_splits  # noqa: E402
from scripts.run_direct_full_asset_evaluation import _populations  # noqa: E402
from scripts.run_sharded_mp_external import _district_external  # noqa: E402
from evaluation.references import madhya_pradesh_districts  # noqa: E402
from scripts.run_seasonal_evaluation import _collapsed_external_predictions  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "development_tuning.json"
LABELS = list(REFERENCE_LABELS.values())
SCHEMA_SLUG = TRAINING_SCHEMA_VERSION.replace("-", "_")


def training_asset_id(season):
    return f"{EE_ASSET_ROOT}/tuning_{SCHEMA_SLUG}_{season}_train"


def materialised_training_table(season, retries):
    """Sample the training points once into an asset, then train from that.

    Twenty-eight classifiers trained from a live sampleRegions means Earth
    Engine rebuilds the composite, the indices, the texture and the radar for
    every one of them, on every district -- the first district took twenty-five
    minutes. Exporting the sampled table once turns that into a table read, and
    the same fixed rows then feed every configuration, which is also what makes
    the comparison fair.
    """
    asset = training_asset_id(season)

    def existing():
        try:
            return ee.FeatureCollection(asset).size().getInfo()
        except ee.EEException:
            return None

    size = existing()
    if size:
        print(f"reusing {asset} ({size} rows)", flush=True)
        return ee.FeatureCollection(asset)

    dates = SEASONS[season]
    table = _call_with_retry(
        f"{season}-sample",
        lambda: sample_training_points(
            _populations()["after"],
            start_date=dates["start"],
            end_date=dates["end"],
            bands=FULL_BANDS,
        ),
        retries,
    )
    task = ee.batch.Export.table.toAsset(
        collection=table,
        description=f"tuning_{season}_train",
        assetId=asset,
    )
    # Four workers reaching this at once all try to create the same asset, and
    # Earth Engine fails every attempt after the first with "Cannot overwrite
    # asset". That is a race, not an error: the loser's job is to wait for the
    # winner's export and then read it, which is what the poll below does.
    task.start()
    print(f"exporting {asset} ...", flush=True)
    deadline = time.time() + 3600
    while True:
        status = task.status()
        state = status.get("state")
        if state == "COMPLETED":
            break
        if state in ("FAILED", "CANCELLED", "CANCEL_REQUESTED"):
            message = status.get("error_message") or ""
            if "overwrite" not in message.lower():
                raise RuntimeError(f"export {state}: {message}")
            print(f"another worker is exporting {asset}; waiting", flush=True)
            while time.time() < deadline:
                size = existing()
                if size:
                    print(f"reusing {asset} ({size} rows)", flush=True)
                    return ee.FeatureCollection(asset)
                time.sleep(20)
            raise RuntimeError(f"timed out waiting for {asset}")
        if time.time() > deadline:
            raise RuntimeError(f"export of {asset} did not finish in an hour")
        time.sleep(20)
    size = ee.FeatureCollection(asset).size().getInfo()
    print(f"exported {asset} ({size} rows)", flush=True)
    return ee.FeatureCollection(asset)

# maxNodes=None means "leave it unset", which is Earth Engine's unlimited.
GRIDS = {
    "rf": [
        {"numberOfTrees": trees, "bagFraction": 0.5, "maxNodes": nodes,
         "minLeafPopulation": 1}
        for trees in (100, 300)
        for nodes in (10, 50, None)
    ],
    "gtb": [
        {"numberOfTrees": trees, "shrinkage": shrinkage, "maxNodes": nodes}
        for trees in (100, 300)
        for shrinkage in (0.05, 0.1)
        for nodes in (10, 50)
    ],
    "svm": [
        {"kernelType": "RBF", "gamma": gamma, "cost": cost}
        for gamma in (0.1, 0.5, 1.0)
        for cost in (10.0, 100.0)
    ],
    "cart": [{"maxNodes": nodes} for nodes in (10, 20, 50, None)],
    "knn": [{"k": k} for k in (3, 5, 9, 15)],
}

# Cumulative feature families, for the attribution the paper reports as
# Table VIII. Those numbers -- "open-land recall from 7% to 54%" among them --
# were measured under an older four-class WorldCover protocol on a pooled
# sample, so they cannot be quoted beside v4 results without re-measuring.
# Attribution is a development activity, so it runs on the development
# districts and never touches the test half.
GROUP_STACKS = {
    "g10_baseline": ["B2", "B3", "B4", "B8", "B11", "B12",
                     "NDVI", "NDWI", "NDBI", "SAVI"],
    "g15_bareness": ["B2", "B3", "B4", "B8", "B11", "B12",
                     "NDVI", "NDWI", "NDBI", "SAVI",
                     "BSI", "UI", "IBI", "SWIRratio", "BAEI"],
    "g21_texture": ["B2", "B3", "B4", "B8", "B11", "B12",
                    "NDVI", "NDWI", "NDBI", "SAVI",
                    "BSI", "UI", "IBI", "SWIRratio", "BAEI",
                    "g_contrast", "g_ent", "g_var", "g_idm", "g_diss",
                    "g_asm"],
    "g27_sar": list(FULL_BANDS),
}
GROUP_MODELS = ("rf", "gtb")

assert len(GROUP_STACKS["g10_baseline"]) == 10
assert len(GROUP_STACKS["g27_sar"]) == 27
assert all(set(stack) <= set(FULL_BANDS) for stack in GROUP_STACKS.values())

# Leave-one-band-out over the 27-band stack. Removal rather than addition,
# because a forward-additive table can only ever show a feature helping: a band
# that actively costs accuracy is still present in every later row. A removal
# that raises accuracy is a band that was hurting.
ABLATION_MODEL = "gtb"

def _config_id(model, params):
    parts = "_".join(f"{key}{value}" for key, value in sorted(params.items()))
    return f"{model}:{parts}"


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
        except RETRYABLE_ERRORS as caught:
            error = caught
            print(f"retry {label} attempt={attempt}: {caught}", flush=True)
            time.sleep(10 * attempt)
    raise error


def _evaluate(external, classifiers, keys):
    payload = {"sample_count": external.size()}
    for key in keys:
        classified = _collapsed_external_predictions(
            external.classify(classifiers[key]), "after")
        payload[key] = classified.errorMatrix(
            "reference", "external_prediction", list(REFERENCE_LABELS)
        ).array().toList()
    return ee.Dictionary(payload).getInfo()


def _evaluate_resilient(external, classifiers, keys, retries, label):
    try:
        return _call_with_retry(label, lambda: _evaluate(
            external, classifiers, keys), retries)
    except RETRYABLE_ERRORS:
        if len(keys) == 1:
            raise
        midpoint = len(keys) // 2
        print(f"split {label} configs={len(keys)}", flush=True)
        left = _evaluate_resilient(
            external, classifiers, keys[:midpoint], retries, label)
        right = _evaluate_resilient(
            external, classifiers, keys[midpoint:], retries, label)
        return {**left, **right}


def run(args):
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    _call_with_retry("initialize", init_ee, args.retries)
    splits = district_splits()
    development = splits["development"]
    every = splits[args.districts]
    # Same striping as scripts/run_sharded_mp_external.py: districts are dealt
    # round-robin so each worker gets a mix of large and small ones rather than
    # one worker inheriting every big district in an alphabetical block.
    chosen = [name for index, name in enumerate(every)
              if index % args.worker_count == args.worker_index]
    populations = _populations()
    leakage = populations["before"].merge(populations["after"])
    districts = madhya_pradesh_districts()

    output = args.output
    state = json.loads(output.read_text()) if output.exists() else {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "protocol": (
            "Development and test districts are disjoint and were partitioned "
            "before anything was selected. A search runs on development; a "
            "frozen choice is scored once on test."
        ),
        "districts_used": args.districts,
        "districts": every,
        "assigned_districts": chosen,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "mode": args.mode,
        "shards": {},
    }

    stacks = dict(GROUP_STACKS)
    if args.mode == "groups":
        configurations = [(model, {"_stack": stack})
                          for model in GROUP_MODELS
                          for stack in GROUP_STACKS]
    elif args.mode == "ablation":
        # The full stack, then the full stack minus each band in turn. Both
        # families are scored in the same pass as the removals, so the
        # attribution table and the ablation come from identical pixels.
        stacks = {"all27": list(FULL_BANDS)}
        for band in FULL_BANDS:
            stacks[f"drop_{band}"] = [b for b in FULL_BANDS if b != band]
        stacks.update(GROUP_STACKS)
        configurations = [(ABLATION_MODEL, {"_stack": name}) for name in stacks]
    elif args.mode == "stacks":
        if not args.stacks:
            raise SystemExit("--mode stacks needs --stacks pointing at a JSON "
                             "object of {name: [bands]}")
        stacks = json.loads(Path(args.stacks).read_text())
        stacks = stacks.get("stacks", stacks)
        configurations = [(model, {"_stack": name})
                          for model in args.models.split(",")
                          for name in stacks]
    else:
        configurations = [
            (model, params) for model, grid in GRIDS.items() for params in grid]
    print(f"{len(configurations)} configurations x {len(chosen)} "
          f"{args.districts} districts, season={args.season}, "
          f"mode={args.mode}", flush=True)

    training = materialised_training_table(args.season, args.retries)
    classifiers = {}
    for model, params in configurations:
        stack = params.get("_stack")
        # A hyperparameter search varies the model, so it uses whatever stack
        # is currently in production rather than a pinned historical one.
        bands = stacks[stack] if stack else BANDS
        settings = {key: value for key, value in params.items()
                    if key != "_stack"}
        key = f"{model}:{stack}" if stack else _config_id(model, settings)
        # A stack experiment varies the features, not the model, so it uses the
        # shipped canonical settings rather than an empty parameter set --
        # ee.Classifier.smileGradientTreeBoost has required arguments and would
        # otherwise fail before the first request went out.
        estimator = make_classifier(model, settings or None)
        classifiers[key] = estimator.train(
            features=training,
            classProperty="label",
            inputProperties=bands,
        )
    keys = list(classifiers)

    for name in chosen:
        shard_id = f"{args.season}:{name}"
        if shard_id in state["shards"]:
            print(f"skip {shard_id}", flush=True)
            continue
        district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
        external = _district_external(district, name, args.season, leakage)
        payload = {}
        # Batched so one district is several moderate requests rather than one
        # request Earth Engine will refuse outright.
        for start in range(0, len(keys), args.batch):
            batch = keys[start:start + args.batch]
            print(f"evaluate {shard_id} configs {start}-{start + len(batch)}",
                  flush=True)
            payload.update(_evaluate_resilient(
                external, classifiers, batch, args.retries, shard_id))
        state["shards"][shard_id] = {
            "season": args.season,
            "district": name,
            "sample_count": payload.pop("sample_count"),
            "matrices": payload,
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output, state)
        print(f"landed {shard_id}", flush=True)

    # A worker sees only its own districts, so summarising here would report a
    # third of the evidence as if it were all of it. The merge step does it.
    if args.worker_count == 1:
        summarise(state, args.season)
    _write_json(output, state)
    print(f"wrote {output}", flush=True)


def summarise(state, season):
    shards = [shard for shard in state["shards"].values()
              if shard["season"] == season]
    if not shards:
        return
    size = len(LABELS)
    scores, matrices = {}, {}
    for key in shards[0]["matrices"]:
        matrix = [[0] * size for _ in range(size)]
        for shard in shards:
            value = shard["matrices"][key]
            for row in range(size):
                for column in range(size):
                    matrix[row][column] += value[row][column]
        matrices[key] = matrix
        metrics = metrics_from_matrix(matrix, LABELS)
        scores[key] = {
            "overall_accuracy": metrics["overall_accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "sample_count": metrics["sample_count"],
            "per_class_recall": {
                label: metrics["per_class"][label]["recall"] for label in LABELS
            },
            "per_class_precision": {
                label: metrics["per_class"][label]["precision"]
                for label in LABELS
            },
        }
    state.setdefault("scores", {})[season] = scores
    # Kept because an aggregate metric cannot answer "which error did this
    # feature family leave behind": that needs the off-diagonal cell, and
    # recovering it later would mean re-summing every shard.
    state.setdefault("pooled_matrices", {})[season] = matrices
    best = {}
    for key, value in scores.items():
        model = key.split(":", 1)[0]
        if model not in best or value["overall_accuracy"] > best[model][1]["overall_accuracy"]:
            best[model] = (key, value)
    state.setdefault("best_per_model", {})[season] = {
        model: {"config": key, **value} for model, (key, value) in best.items()}
    where = state.get("districts_used", "development")
    print(f"\n== {where}, {season} ==")
    for model, (key, value) in sorted(
            best.items(), key=lambda item: -item[1][1]["overall_accuracy"]):
        print(f"  {model:5s} best {value['overall_accuracy'] * 100:5.2f}%  "
              f"kappa {value['kappa']:.3f}  {key.split(':', 1)[1]}")

    baseline_key = f"{ABLATION_MODEL}:all27"
    if baseline_key in scores:
        summarise_ablation(state, season, scores, baseline_key)


def summarise_ablation(state, season, scores, baseline_key):
    """Rank each band by what removing it does, and name the bands to drop.

    A band is a candidate for removal when the stack scores at least as well
    without it. That is a weaker test than "removal helps significantly", and
    deliberately so: the question a feature stack has to answer is not whether
    a band helps but whether it earns the compute, and a band that cannot be
    shown to help does not.
    """
    baseline = scores[baseline_key]["overall_accuracy"]
    rows = []
    for key, value in scores.items():
        model, name = key.split(":", 1)
        if not name.startswith("drop_"):
            continue
        band = name[len("drop_"):]
        rows.append({
            "band": band,
            "overall_accuracy_without": value["overall_accuracy"],
            "delta_points": (value["overall_accuracy"] - baseline) * 100,
            "open_land_recall_without": value["per_class_recall"].get("Open Land"),
        })
    rows.sort(key=lambda row: -row["delta_points"])
    droppable = [row["band"] for row in rows if row["delta_points"] >= 0]
    state.setdefault("ablation", {})[season] = {
        "baseline_key": baseline_key,
        "baseline_overall_accuracy": baseline,
        "bands": rows,
        "droppable_bands": droppable,
    }
    print(f"\n  leave-one-out against {baseline_key} "
          f"({baseline * 100:.2f}%), {season}:")
    for row in rows:
        flag = "  drop" if row["delta_points"] >= 0 else ""
        print(f"    -{row['band']:<12s} {row['overall_accuracy_without'] * 100:6.2f}% "
              f"{row['delta_points']:+6.2f}{flag}")
    print(f"  {len(droppable)} bands are not earning their place: "
          f"{', '.join(droppable) or 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter", choices=list(SEASONS))
    parser.add_argument("--mode", default="grid",
                        choices=("grid", "groups", "ablation", "stacks"))
    parser.add_argument("--districts", default="development",
                        choices=("development", "test"))
    parser.add_argument("--stacks", type=Path,
                        help="JSON object of {name: [bands]} for --mode stacks")
    parser.add_argument("--models", default="gtb",
                        help="comma-separated models for --mode stacks")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    if args.summarise_only:
        state = json.loads(args.output.read_text())
        summarise(state, args.season)
        _write_json(args.output, state)
        return
    run(args)


if __name__ == "__main__":
    main()
