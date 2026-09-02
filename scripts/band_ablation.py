"""Leave-one-out band ablation for the 27-band feature stack.

The existing attribution table in doc/mp_accuracy_report.md is forward-additive
(BASE, +bareness, +texture, +SAR), which can only ever show a group helping.
It cannot show a band that is actively costing accuracy, because a band that
hurts is still present in every later row.

This runs the other direction: train on the full stack, then retrain with one
band (or one group) removed and measure what happens. A removal that *raises*
accuracy is a band that was hurting.

Two phases, because the expensive part is compositing Madhya Pradesh:

    python scripts/band_ablation.py sample   # ~10 min, exports two EE assets
    python scripts/band_ablation.py run      # ~15 min, writes results JSON

`sample` materialises the training table and the MP evaluation table as EE
assets. `run` then retrains against those fixed tables, so every configuration
sees identical pixels and the only variable is the band list.
"""

import argparse
import json
import os
import socket
import sys
import time

import ee

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import BANDS as PRODUCTION_BANDS
from backend.config import EE_ASSET_ROOT, FEATURE_COLLECTIONS
from backend.gee_classifier import (
    _add_spectral_indices, add_sar_bands, init_ee, mask_s2_clouds)

START_DATE = "2025-03-01"
END_DATE = "2025-04-30"
CLOUD_THRESHOLD = 15

# Without this a dropped connection blocks getInfo forever: the process stays
# alive, stops logging, and never finishes. retry() only helps once a call
# actually returns an error. run_sharded_mp_external.py sets the same timeout
# for the same reason.
socket.setdefaulttimeout(600)

TRAIN_ASSET = f"{EE_ASSET_ROOT}/ablation_train_table"
EVAL_ASSET = f"{EE_ASSET_ROOT}/ablation_mp_eval_table"
JAB_EVAL_ASSET = f"{EE_ASSET_ROOT}/ablation_jabalpur_eval_table"

# Jabalpur district, where 4,000 of the 5,000 training points live. Evaluating
# here answers a different question from the state-wide run: not "does this
# generalise" but "is this band earning its place on home ground".
# The district spans 79.35-80.58 E; splitting on the midpoint gives two disjoint
# halves, so bands can be chosen on one and the choice tested on the other.
JAB_SPLIT_LON = 79.961
JAB_POINTS_PER_CLASS = 400

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "doc", "assets",
                            "band_ablation_results.json")

# ESA WorldCover -> project classes, only where the mapping is unambiguous.
# Cropland (40) is deliberately absent: in a March-April composite a field may be
# bare or green, so it has no honest mapping onto four classes.
WORLDCOVER_MAP = {10: 0, 80: 1, 50: 2, 60: 3}
CLASS_NAMES = ["Vegetation", "Water", "Built Area", "Open Land"]

# The same 4x3 tiling the MP accuracy study used: a single state-wide request
# exceeds the interactive compute limit.
MP_BOUNDS = (74.0, 21.0, 82.8, 26.9)
TILE_COLS, TILE_ROWS = 4, 3
POINTS_PER_CLASS_PER_TILE = 120

GROUPS = {
    "Optical reflectance": ["B2", "B3", "B4", "B8", "B11", "B12"],
    "Core indices": ["NDVI", "NDWI", "NDBI", "SAVI"],
    "Bareness indices": ["BSI", "UI", "IBI", "SWIRratio", "BAEI"],
    "Optical GLCM texture": ["g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm"],
    "SAR amplitude": ["VV", "VH", "VVVH"],
    "SAR GLCM texture": ["s_contrast", "s_var", "s_ent"],
}

# The 27-band stack this ablation exists to test, in composite order.
#
# This was backend.config.BANDS when the study was written. config has since
# been cut to the 19 bands the study selected, so importing it would leave every
# entry in STACKS dropping bands that are no longer present, all three "stacks"
# identical, and the ablation a silent no-op that reproduces its own old numbers.
# The assertions below are the check that would have caught that.
BANDS = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "NDVI", "NDWI", "NDBI", "SAVI",
    "BSI", "UI", "IBI", "SWIRratio", "BAEI",
    "g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm",
    "VV", "VH", "VVVH", "s_contrast", "s_var", "s_ent",
]
assert len(BANDS) == 27, len(BANDS)
assert set(BANDS) == {b for members in GROUPS.values() for b in members}
assert set(PRODUCTION_BANDS) < set(BANDS)

SEEDS = [1, 2, 3]

# The shipped forest is capped at maxNodes=10. That cap is itself a confound: a
# tree with ten nodes has ten chances to use a feature, so "this band adds
# nothing" and "this forest is too shallow to use it" look identical. Re-running
# the ablation uncapped (`--max-nodes 0`) separates them.
MAX_NODES = 10

MODEL = "rf"

_COUNTS = {}


# ── phase 1: sampling ────────────────────────────────────────────────────────

def retry(fn, tries=7):
    """Earth Engine's non-commercial tier throttles concurrent compute requests.

    A 429 here means "come back later", not "this query is wrong", so back off
    and repeat rather than losing a half-hour of compositing.
    """
    for attempt in range(tries):
        try:
            return fn()
        except ee.EEException as exc:
            transient = "Too Many Requests" in str(exc) or "concurrency" in str(exc)
            if not transient or attempt == tries - 1:
                raise
            delay = 15 * 2 ** attempt
            print(f"  throttled, retrying in {delay}s", flush=True)
            time.sleep(delay)


def _composite(geometry, clip):
    """The production composite from backend/gee_classifier.py, minus its probes.

    `_build_collection` and `_add_sar` each call `.size().getInfo()` to decide
    whether to widen an empty date window. Twelve tiles would spend 24 throttled
    round-trips on a question already answered: the March-April 2025 window has
    both Sentinel-2 and Sentinel-1 coverage over every tile of Madhya Pradesh.
    Everything else -- the cloud mask, the median, the index and texture stack,
    the dB rescaling and the 0-1 band rescaling -- is the shipped code, called
    rather than copied. The SAR half used to be a copy, and it silently missed
    the texture rescaling when that was added.
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
    )
    composite = s2.median()
    if clip:
        composite = composite.clip(geometry)
    composite = _add_spectral_indices(composite)

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    return add_sar_bands(composite, s1.select(["VV", "VH"]).median())


def build_training_table():
    collections = [ee.FeatureCollection(p) for p in FEATURE_COLLECTIONS.values()]
    all_points = collections[0]
    for fc in collections[1:]:
        all_points = all_points.merge(fc)

    geometry = all_points.geometry().bounds()
    composite = _composite(geometry, clip=False)
    return (
        composite.select(BANDS)
        # geometries=True: a table asset cannot hold features with null geometry.
        .sampleRegions(collection=all_points, properties=["label"], scale=10,
                       tileScale=4, geometries=True)
        .filter(ee.Filter.notNull(BANDS + ["label"]))
    )


def build_eval_tile_tables():
    """One stratified sample table per in-state tile, labelled by ESA WorldCover.

    Returned per tile rather than merged, because compositing all twelve tiles
    inside a single export exceeds the Earth Engine batch compute limit and the
    job dies with "Computation timed out". Each tile exports on its own and a
    second, cheap export merges the finished shards.
    """
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    reference = (
        worldcover.remap(list(WORLDCOVER_MAP), list(WORLDCOVER_MAP.values()))
        .rename("reference")
    )

    lon0, lat0, lon1, lat1 = MP_BOUNDS
    dx = (lon1 - lon0) / TILE_COLS
    dy = (lat1 - lat0) / TILE_ROWS
    mp = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq("ADM1_NAME", "Madhya Pradesh"))
        .geometry()
    )

    cells = [(c, r) for c in range(TILE_COLS) for r in range(TILE_ROWS)]
    regions = {
        (c, r): ee.Geometry.Rectangle([
            lon0 + c * dx, lat0 + r * dy, lon0 + (c + 1) * dx, lat0 + (r + 1) * dy,
        ]).intersection(mp, maxError=100)
        for c, r in cells
    }
    # One request for all twelve areas: corner tiles fall outside the state and
    # sampling them would only burn throttled compute.
    areas = retry(lambda: ee.List(
        [regions[cell].area(maxError=100) for cell in cells]).getInfo())

    tables = []
    for (col, row), area in zip(cells, areas):
        if area < 1e6:
            print(f"  tile c{col}r{row} outside MP, skipped", flush=True)
        else:
            region = regions[(col, row)]
            print(f"  tile c{col}r{row} ...", flush=True)
            stack = _composite(region, clip=True).select(BANDS).addBands(reference)
            samples = stack.stratifiedSample(
                numPoints=POINTS_PER_CLASS_PER_TILE,
                classBand="reference",
                region=region,
                scale=10,
                seed=col * 10 + row,
                tileScale=8,
                geometries=True,
            )
            tables.append(((col, row), samples.map(lambda f: f.set("tile_col", col))))

    return tables


def build_jabalpur_eval_table():
    """Stratified points inside Jabalpur district, tagged east or west.

    One region, not a tiling: the district is 4,021 km2 against the state's
    308,776, so a single composite stays well inside the compute limit.
    """
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    reference = (
        worldcover.remap(list(WORLDCOVER_MAP), list(WORLDCOVER_MAP.values()))
        .rename("reference")
    )
    region = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM2_NAME", "Jabalpur"))
        .geometry()
    )
    stack = _composite(region, clip=True).select(BANDS).addBands(reference)
    samples = stack.stratifiedSample(
        numPoints=JAB_POINTS_PER_CLASS,
        classBand="reference",
        region=region,
        scale=10,
        seed=42,
        tileScale=8,
        geometries=True,
    )
    # Longitude travels with the point so the halves can be split at query time
    # without re-sampling.
    samples = samples.map(lambda f: f.set("lon", f.geometry().coordinates().get(0)))
    return samples.filter(ee.Filter.notNull(BANDS + ["reference"]))


def export_to_asset(table, asset_id, description):
    try:
        ee.data.deleteAsset(asset_id)
        print(f"deleted existing {asset_id}")
    except Exception:
        pass
    task = ee.batch.Export.table.toAsset(
        collection=table, description=description, assetId=asset_id)
    task.start()
    print(f"export started: {description} -> {asset_id}")
    return task


def wait_for(tasks):
    pending = {t.id: t for t in tasks}
    seen = {}
    while pending:
        time.sleep(30)
        for task_id, task in list(pending.items()):
            # Even asking "is it done yet" counts against the concurrency limit.
            status = retry(task.status)
            state, name = status["state"], task.config.get("description")
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                print(f"{name}: {state}", flush=True)
                if state != "COMPLETED":
                    print(status.get("error_message"), flush=True)
                del pending[task_id]
            elif seen.get(task_id) != state:
                seen[task_id] = state
                print(f"{name}: {state}", flush=True)


def cmd_sample(args):
    init_ee()
    if args.region in ("mp", "all"):
        if args.skip_train:
            print("training table: skipped")
        else:
            print("building training table ...")
            wait_for([export_to_asset(
                build_training_table(), TRAIN_ASSET, "ablation_train_table")])
        print("building MP evaluation shards ...")
        shards, tasks = [], []
        for (col, row), table in build_eval_tile_tables():
            asset = f"{EVAL_ASSET}_c{col}r{row}"
            shards.append(asset)
            tasks.append(export_to_asset(
                table, asset, f"ablation_mp_eval_c{col}r{row}"))
        wait_for(tasks)
        print(f"merging {len(shards)} evaluation shards ...")
        merged = ee.FeatureCollection(shards[0])
        for asset in shards[1:]:
            merged = merged.merge(ee.FeatureCollection(asset))
        wait_for([export_to_asset(
            merged.filter(ee.Filter.notNull(BANDS + ["reference"])),
            EVAL_ASSET, "ablation_mp_eval_table")])
    if args.region in ("jabalpur", "all"):
        print("building Jabalpur evaluation table ...")
        wait_for([export_to_asset(
            build_jabalpur_eval_table(), JAB_EVAL_ASSET,
            "ablation_jabalpur_eval_table")])


# ── phase 2: ablation ────────────────────────────────────────────────────────

def make_classifier(seed):
    """The requested model with the exact hyperparameters the application ships.

    Only Random Forest and Gradient Boost take a seed. SVM, CART and KNN are
    deterministic, so for those a configuration is measured once and the delta
    carries no model-sampling noise at all.
    """
    if MODEL == "rf":
        params = dict(numberOfTrees=200, minLeafPopulation=1, bagFraction=0.3, seed=seed)
        if MAX_NODES:
            params["maxNodes"] = MAX_NODES
        return ee.Classifier.smileRandomForest(**params)
    if MODEL in ("gtb", "xgb"):
        params = dict(numberOfTrees=100, shrinkage=0.1, seed=seed)
        if MAX_NODES:
            params["maxNodes"] = MAX_NODES
        return ee.Classifier.smileGradientTreeBoost(**params)
    if MODEL == "svm":
        return ee.Classifier.libsvm(kernelType="RBF", gamma=1.0, cost=100.0)
    if MODEL == "cart":
        return ee.Classifier.smileCart(maxNodes=20)
    if MODEL == "knn":
        return ee.Classifier.smileKNN(k=5)
    raise ValueError(f"unknown model {MODEL}")


DETERMINISTIC = {"svm", "cart", "knn"}


def evaluate(train_fc, eval_fc, bands, seed):
    """Train the shipped classifier on `bands` only and score it against the MP table."""
    classifier = make_classifier(seed).train(
        features=train_fc, classProperty="label", inputProperties=bands)
    classified = eval_fc.classify(classifier)
    matrix = classified.errorMatrix("reference", "classification", [0, 1, 2, 3])
    return ee.Dictionary({
        "overall": matrix.accuracy(),
        "kappa": matrix.kappa(),
        "recall": matrix.producersAccuracy().toList().flatten(),
        "precision": matrix.consumersAccuracy().toList().flatten(),
        "soil_to_built": matrix.array().get([3, 2]),
        "matrix": matrix.array().toList(),
    })


def run_config(train_fc, eval_fc, bands):
    """Mean over seeds: one forest is noisy enough on its own to fake a finding."""
    results = [evaluate(train_fc, eval_fc, bands, s) for s in SEEDS]
    raw = retry(lambda: ee.List(results).getInfo())
    n = len(raw)
    overalls = [r["overall"] for r in raw]
    mean = sum(overalls) / n
    # Standard error, not max-minus-min: spread grows with the number of seeds,
    # so it would punish the very runs that measure the mean most precisely.
    var = sum((o - mean) ** 2 for o in overalls) / (n - 1) if n > 1 else 0.0
    return {
        "overall": mean,
        "kappa": sum(r["kappa"] for r in raw) / n,
        "recall": [sum(r["recall"][i] for r in raw) / n for i in range(4)],
        "precision": [sum(r["precision"][i] for r in raw) / n for i in range(4)],
        "soil_to_built": sum(r["soil_to_built"] for r in raw) / n,
        "matrix": [[sum(r["matrix"][i][j] for r in raw) / n for j in range(4)]
                   for i in range(4)],
        "spread": max(overalls) - min(overalls),
        "sem": (var / n) ** 0.5,
        "overall_seeds": overalls,
    }


def cmd_run(args):
    global MAX_NODES, RESULTS_PATH, SEEDS, MODEL
    MODEL = args.model
    MAX_NODES = args.max_nodes
    SEEDS = [1] if MODEL in DETERMINISTIC else list(range(1, args.seeds + 1))
    suffix = "" if MODEL == "rf" else f"_{MODEL}"
    if not MAX_NODES and MODEL in ("rf", "gtb", "xgb"):
        suffix += "_deep"
    RESULTS_PATH = RESULTS_PATH.replace(".json", f"{suffix}.json")
    print(f"model: {MODEL}   maxNodes: {MAX_NODES or 'unlimited'}   seeds: {len(SEEDS)}")
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    eval_fc = ee.FeatureCollection(EVAL_ASSET)
    _COUNTS["n_train"] = train_fc.size().getInfo()
    _COUNTS["n_eval"] = eval_fc.size().getInfo()
    print(f"train points: {_COUNTS['n_train']}   eval points: {_COUNTS['n_eval']}")

    configs = [("ALL", "baseline", BANDS)]
    for name, members in GROUPS.items():
        configs.append((f"-{name}", "group",
                        [b for b in BANDS if b not in members]))
    for band in BANDS:
        configs.append((f"-{band}", "band", [b for b in BANDS if b != band]))

    results = []
    for label, kind, bands in configs:
        started = time.time()
        metrics = run_config(train_fc, eval_fc, bands)
        metrics.update(label=label, kind=kind, n_bands=len(bands),
                       removed=[b for b in BANDS if b not in bands])
        results.append(metrics)
        print(f"{label:24s} {len(bands):3d} bands  "
              f"overall {metrics['overall'] * 100:5.2f}%  "
              f"soil recall {metrics['recall'][3] * 100:5.2f}%  "
              f"({time.time() - started:.0f}s)", flush=True)
        _save(results)

    # Leave-one-out cannot see redundancy: two bands carrying the same signal both
    # score ~0 because each covers for the other. Dropping every band that was not
    # individually load-bearing tests the combination the single-band runs imply.
    baseline = results[0]["overall"]
    dead = [r["removed"][0] for r in results
            if r["kind"] == "band" and r["overall"] >= baseline]
    if dead:
        keep = [b for b in BANDS if b not in dead]
        metrics = run_config(train_fc, eval_fc, keep)
        metrics.update(label="-all non-load-bearing", kind="combined",
                       n_bands=len(keep), removed=dead)
        results.append(metrics)
        print(f"{metrics['label']:24s} {len(keep):3d} bands  "
              f"overall {metrics['overall'] * 100:5.2f}%  "
              f"soil recall {metrics['recall'][3] * 100:5.2f}%")
        _save(results)
    print(f"\nwrote {RESULTS_PATH}")


def _save(results):
    payload = {"class_names": CLASS_NAMES, "groups": GROUPS, "bands": BANDS,
               "seeds": SEEDS, "max_nodes": MAX_NODES, "model": MODEL,
               "results": results}
    payload.update(_COUNTS)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)


# ── phase 3: what the cut actually does to Soil vs Buildings ────────────

# Bands chosen individually by the single-band ablation, not by family.
# `cut5` are the bands whose removal gained accuracy in BOTH tree models, clearing
# the 95% interval in at least one. `cut8` widens that to everything significant
# in the nine-seed Random Forest run, which pulls in three raw reflectance bands
# that Gradient Boost does not agree about.
STACKS = {
    "shipped": [],
    "cut5": ["B8", "NDVI", "NDWI", "NDBI", "s_ent"],
    "cut8": ["B2", "B4", "B8", "NDVI", "NDWI", "NDBI", "SAVI", "s_ent"],
}


def cmd_confusion(args):
    """Full confusion matrices for the shipped stack and the two proposed cuts.

    The ablation answered "does overall accuracy move". This answers the question
    the project actually cares about: where do the bare-soil points end up.
    """
    global MODEL, SEEDS, MAX_NODES
    MAX_NODES = 10
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    eval_fc = ee.FeatureCollection(EVAL_ASSET)

    out = {"class_names": CLASS_NAMES, "bands": BANDS, "stacks": STACKS,
           "n_train": train_fc.size().getInfo(), "n_eval": eval_fc.size().getInfo(),
           "models": {}}
    print(f"train {out['n_train']}  eval {out['n_eval']}")

    for model, seeds in args.models:
        MODEL, SEEDS = model, [1] if model in DETERMINISTIC else list(range(1, seeds + 1))
        out["models"][model] = {"seeds": len(SEEDS), "stacks": {}}
        for name, dropped in STACKS.items():
            bands = [b for b in BANDS if b not in dropped]
            m = run_config(train_fc, eval_fc, bands)
            m.update(dropped=dropped, n_bands=len(bands))
            out["models"][model]["stacks"][name] = m
            print(f"{model:4s} {name:12s} {len(bands):3d} bands  "
                  f"overall {m['overall'] * 100:5.2f}%  "
                  f"soil recall {m['recall'][3] * 100:5.1f}%  "
                  f"built prec {m['precision'][2] * 100:5.1f}%  "
                  f"soil->built {m['soil_to_built']:5.0f}", flush=True)

    path = os.path.join(os.path.dirname(__file__), "..", "doc", "assets",
                        "mp_band_cut_confusion.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


# ── phase 4: individual-band selection on Jabalpur, validated out of sample ──

def cmd_jabalpur(args):
    """Pick the bands that cost accuracy over Jabalpur, then test the pick.

    Choosing bands by their score on a set and then reporting that same score is
    how ablations flatter themselves. The district splits cleanly at 79.961 E, so
    selection runs on the west half and the resulting stack is scored on the east
    half, which had no say in choosing it.
    """
    global MODEL, SEEDS, MAX_NODES
    MAX_NODES = 10
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    full = ee.FeatureCollection(JAB_EVAL_ASSET)
    west = full.filter(ee.Filter.lt("lon", JAB_SPLIT_LON))
    east = full.filter(ee.Filter.gte("lon", JAB_SPLIT_LON))

    halves = {"west": west, "east": east, "full": full}
    counts = retry(lambda: ee.Dictionary(
        {k: v.size() for k, v in halves.items()}).getInfo())
    print(f"Jabalpur eval points: {counts}")

    out = {"class_names": CLASS_NAMES, "bands": BANDS, "groups": GROUPS,
           "split_lon": JAB_SPLIT_LON, "counts": counts,
           "n_train": train_fc.size().getInfo(), "select": {}, "validate": {}}

    # ── selection pass: every band, on the west half only ──
    for model, seeds in args.models:
        MODEL, SEEDS = model, [1] if model in DETERMINISTIC else list(range(1, seeds + 1))
        base = run_config(train_fc, west, BANDS)
        rows = {"__baseline__": base}
        print(f"\n[{model}] west baseline {base['overall'] * 100:.2f}%")
        for band in BANDS:
            keep = [b for b in BANDS if b != band]
            m = run_config(train_fc, west, keep)
            rows[band] = m
            d = (m["overall"] - base["overall"]) * 100
            ci = 1.96 * ((m.get("sem", 0) ** 2 + base.get("sem", 0) ** 2) ** 0.5) * 100
            print(f"  −{band:<12s} {d:+6.2f}  ±{ci:.2f}"
                  f"{'  COSTS' if d > ci else ''}", flush=True)
        out["select"][model] = rows

    # ── the cut: bands that cost accuracy in every model asked, ──
    # ── and clear their own interval in at least one of them.    ──
    costly = []
    for band in BANDS:
        deltas, cleared = [], False
        for model, _ in args.models:
            r, b = out["select"][model][band], out["select"][model]["__baseline__"]
            d = (r["overall"] - b["overall"]) * 100
            ci = 1.96 * ((r.get("sem", 0) ** 2 + b.get("sem", 0) ** 2) ** 0.5) * 100
            deltas.append(d)
            cleared = cleared or d > ci
        if all(d > 0 for d in deltas) and cleared:
            costly.append(band)
    out["costly"] = costly
    kept = [b for b in BANDS if b not in costly]
    print(f"\ncostly on the west half ({len(costly)}): {', '.join(costly) or 'none'}")
    print(f"keeping {len(kept)} bands")

    # ── validation pass: on the east half, which chose nothing ──
    for model, seeds in args.models:
        MODEL, SEEDS = model, [1] if model in DETERMINISTIC else list(range(1, seeds + 1))
        out["validate"][model] = {}
        for half in ("east", "full"):
            for name, bands in (("shipped", BANDS), ("cut", kept)):
                m = run_config(train_fc, halves[half], bands)
                m["n_bands"] = len(bands)
                out["validate"][model][f"{half}_{name}"] = m
                print(f"[{model}] {half:5s} {name:8s} {len(bands):2d} bands  "
                      f"overall {m['overall'] * 100:5.2f}%  "
                      f"soil recall {m['recall'][3] * 100:5.1f}%  "
                      f"built prec {m['precision'][2] * 100:5.1f}%", flush=True)

    path = os.path.join(os.path.dirname(__file__), "..", "doc", "assets",
                        "jabalpur_band_selection.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


def cmd_holdout(args):
    """Select bands on western MP, score the result on eastern MP.

    `cut5` and `cut8` were chosen using every point they are then reported on,
    which flatters them by an unknown margin. This repeats the whole procedure
    honestly: tiles 0-1 choose the bands, tiles 2-3 never see the choice, and the
    gap between the two numbers is the size of the self-congratulation.
    """
    global MODEL, SEEDS, MAX_NODES
    MAX_NODES = 10
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    full = ee.FeatureCollection(EVAL_ASSET)
    west = full.filter(ee.Filter.lte("tile_col", 1))
    east = full.filter(ee.Filter.gte("tile_col", 2))
    counts = retry(lambda: ee.Dictionary({"west": west.size(), "east": east.size()}).getInfo())
    print(f"west {counts['west']}  east {counts['east']}")

    MODEL, SEEDS = "rf", list(range(1, 10))
    base = run_config(train_fc, west, BANDS)
    print(f"west baseline {base['overall'] * 100:.2f}%")
    picks, rows = [], {}
    for band in BANDS:
        m = run_config(train_fc, west, [b for b in BANDS if b != band])
        d = (m["overall"] - base["overall"]) * 100
        ci = 1.96 * ((m["sem"] ** 2 + base["sem"] ** 2) ** 0.5) * 100
        rows[band] = {"delta": d, "ci": ci}
        if d > ci:
            picks.append(band)
        print(f"  −{band:<12s} {d:+6.2f} ±{ci:.2f}{'  CUT' if d > ci else ''}", flush=True)

    kept = [b for b in BANDS if b not in picks]
    print(f"\nchosen on west ({len(picks)}): {', '.join(picks) or 'none'}")

    out = {"counts": counts, "west_deltas": rows, "picked_on_west": picks,
           "class_names": CLASS_NAMES, "east": {}}
    for name, bands in (("shipped", BANDS), ("west_pick", kept),
                        ("cut5", [b for b in BANDS if b not in STACKS["cut5"]]),
                        ("cut8", [b for b in BANDS if b not in STACKS["cut8"]])):
        m = run_config(train_fc, east, bands)
        m["n_bands"] = len(bands)
        out["east"][name] = m
        print(f"east  {name:10s} {len(bands):2d} bands  overall {m['overall'] * 100:5.2f}%  "
              f"soil recall {m['recall'][3] * 100:5.1f}%", flush=True)

    path = os.path.join(os.path.dirname(__file__), "..", "doc", "assets",
                        "mp_holdout_check.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    smp = sub.add_parser("sample")
    smp.add_argument("--region", default="mp", choices=["mp", "jabalpur", "all"])
    smp.add_argument("--skip-train", action="store_true",
                     help="Reuse an existing ablation_train_table instead of re-exporting it.")
    smp.set_defaults(func=cmd_sample)
    run = sub.add_parser("run")
    run.add_argument("--max-nodes", type=int, default=10,
                     help="Random Forest maxNodes; 0 for unlimited. "
                          "Default 10 matches what the application ships.")
    run.add_argument("--model", default="rf", choices=["rf", "gtb", "svm", "cart", "knn"],
                     help="Which of the application's five classifiers to ablate.")
    run.add_argument("--seeds", type=int, default=3,
                     help="Seeds averaged per configuration. More seeds shrink the "
                          "standard error, which is what a small delta is measured against.")
    run.set_defaults(func=cmd_run)
    conf = sub.add_parser("confusion")
    conf.set_defaults(func=cmd_confusion,
                      models=[("rf", 9), ("gtb", 5), ("svm", 1)])
    sub.add_parser("holdout").set_defaults(func=cmd_holdout)
    jab = sub.add_parser("jabalpur")
    jab.set_defaults(func=cmd_jabalpur, models=[("rf", 9), ("gtb", 5)])
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
