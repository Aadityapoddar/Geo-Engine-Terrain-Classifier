"""Per-district, per-model evaluation of the five-class stack across Madhya Pradesh.

Adds Sand (label 4) to the four existing classes and measures every one of the
application's five classifiers against an independent reference, district by
district, so the answer to "which model should this district use" stops being a
single state-wide average.

The reference problem, and what is done about it
------------------------------------------------
ESA WorldCover has no sand class. River sand and dry upland soil both fall in
class 60, bare/sparse vegetation, so WorldCover alone cannot score a Sand class
at all. Two independent published layers can split it:

    GRWL water mask (Allen & Pavelsky 2018), DN 255 = river channel
    JRC Global Surface Water v1.4, seasonality 1-11 months = inundated part of
    the year, therefore exposed the rest of it

A bare pixel inside that channel envelope is river sand; a bare pixel outside it
is soil. That is the same definition Arora et al. (2019) used for their own sand
classes -- "sand deposits within river channel" -- so channel membership is the
definition rather than a workaround for the absence of one.

The proxy is honest about what it is not. It calls in-channel bare rock and
dried mud "sand". It misses sand outside the historical channel envelope. GRWL
is a 2018 epoch and GSW runs to 2021, so migrated channels are misplaced, and
both are 30 m against a 10 m grid. Results are therefore reported as agreement
with a channel-constrained bare mask, never as sand accuracy.

Ephemeral rivers are its clearest failure: over the Luni at Samdari the envelope
resolves 0.1 km2, because GRWL maps mean-annual-discharge channels and a river
that is dry most of the year barely registers. That site is scored against the
published areas in Arora et al. instead, in `sites`.

Usage
-----
    python scripts/district_eval.py sample   # exports two EE assets, slow
    python scripts/district_eval.py run      # scores 5 models x 10 districts
    python scripts/district_eval.py sites    # Thar dunes + Luni, descriptive
"""

import argparse
import json
import os
import sys
import time

import ee

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import BANDS, EE_ASSET_ROOT, FEATURE_COLLECTIONS, LAND_COVER_CLASSES
from backend.gee_classifier import init_ee
from band_ablation import _composite, export_to_asset, retry, wait_for

START_DATE = "2025-03-01"
END_DATE = "2025-04-30"

TRAIN_ASSET = f"{EE_ASSET_ROOT}/district_train_table"
EVAL_ASSET = f"{EE_ASSET_ROOT}/district_eval_table"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "doc", "assets")
RESULTS_PATH = os.path.join(OUT_DIR, "district_eval_results.json")
SITES_PATH = os.path.join(OUT_DIR, "district_eval_sites.json")

# Chosen to span the failure geography the MP study identified rather than to
# maximise area: Chambal ravines, arid northwest, Malwa, Bundelkhand, the Narmada
# valley, dense eastern forest, two urban districts, and Jabalpur as the
# training-area control that should score near its 95% internal figure.
DISTRICTS = [
    "Morena", "Sheopur", "Dhar", "Indore", "Bhopal",
    "Chhatarpur", "Hoshangabad", "Jabalpur", "Balaghat", "Rewa",
]

CLASS_IDS = sorted(LAND_COVER_CLASSES)
CLASS_NAMES = [LAND_COVER_CLASSES[i]["name"] for i in CLASS_IDS]

MODELS = ["rf", "svm", "gtb", "cart", "knn"]

POINTS_PER_CLASS = 100

# Descriptive sites, each with an explicit reduction scale.
#
# The scale is pinned per site rather than left to bestEffort. Thar at 30 m is
# 5.9e9 pixels against a 1e9 cap, so bestEffort would silently coarsen the grid
# to whatever fits and the number would change between runs and between sites.
# A stated scale that fits is reproducible; a silently negotiated one is not.
SITE_SCALE = {"Thar dunes (Jaisalmer)": 100, "Luni at Samdari (Arora et al.)": 20}

SITES = {
    # The repo's worst known result: 80.9% Buildings under RF, against WorldCover's
    # 0.4% built. Aeolian dune sand is not river sand, so the channel envelope calls
    # this Soil; the question here is only whether Buildings collapses.
    "Thar dunes (Jaisalmer)": [70.50, 26.80, 71.30, 27.40],
    # Arora et al. (2019) study area, 3157.5 ha. They report 466.7 ha of river sand
    # (67.3 fine + 209.5 coarse + 189.9 wet) at Kappa 0.794, from a field visit.
    "Luni at Samdari (Arora et al.)": [72.5371, 25.7826, 72.6081, 25.8233],
}
ARORA_SAND_HA = 466.7


def districts_fc():
    return ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.And(ee.Filter.eq("ADM1_NAME", "Madhya Pradesh"),
                      ee.Filter.inList("ADM2_NAME", DISTRICTS)))


def reference_image():
    """Five-class reference: WorldCover, with class 60 split by channel membership."""
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    grwl = ee.ImageCollection(
        "projects/sat-io/open-datasets/GRWL/water_mask_v01_01").mosaic()

    seasonal = (gsw.select("seasonality").gte(1)
                .And(gsw.select("seasonality").lte(11)).unmask(0))
    channel = grwl.eq(255).unmask(0).Or(seasonal).Or(gsw.select("max_extent").unmask(0))

    # Cropland (40) stays unmapped: in a March-April composite a field may be bare
    # or green, so it has no honest mapping onto these classes.
    base = worldcover.remap([10, 80, 50, 60], [0, 1, 2, 3]).rename("reference")
    # Bare inside the channel envelope becomes Sand; everything else keeps its class.
    return base.where(base.eq(3).And(channel.eq(1)), 4).rename("reference")


def check_labels():
    """Every training asset must carry its own `label`, because class identity
    travels with the points rather than with the merge order.

    A collection missing `label`, or carrying the wrong value, would merge without
    complaint and train a silently wrong model -- the failure would only surface as
    mediocre accuracy weeks later. Checked once, up front, per asset.
    """
    expected = {name: cid for cid, info in LAND_COVER_CLASSES.items()
                for name in [info["name"].lower()]}
    problems = []
    for key, path in FEATURE_COLLECTIONS.items():
        try:
            hist = retry(lambda: ee.FeatureCollection(path)
                         .aggregate_histogram("label").getInfo())
        except ee.EEException as exc:
            problems.append(f"{key}: unreadable ({str(exc)[:80]})")
            continue
        if not hist:
            problems.append(f"{key} ({path}): no `label` property on its features")
            continue
        # A geometry collection exported straight from the Code Editor has no
        # attributes at all, which arrives here as the literal key "null" rather
        # than as an absent histogram.
        if "null" in hist:
            problems.append(
                f"{key} ({path}): {int(hist['null'])} features have a null `label`. "
                f"Re-export with .map(f => f.set('label', N)).")
            continue
        labels = {int(float(k)) for k in hist}
        stem = key.split("_")[0]
        want = expected.get(stem)
        if want is not None and labels != {want}:
            problems.append(
                f"{key} ({path}): expected label {want} for {stem}, found {sorted(labels)}")
        print(f"  {key:12s} {int(sum(hist.values())):>5,} points  label {sorted(labels)}")
    if problems:
        raise SystemExit("training labels are wrong:\n  " + "\n  ".join(problems))


def build_training_table():
    collections = [ee.FeatureCollection(p) for p in FEATURE_COLLECTIONS.values()]
    points = collections[0]
    for fc in collections[1:]:
        points = points.merge(fc)
    composite = _composite(points.geometry().bounds(), clip=False)
    return (composite.select(BANDS)
            .sampleRegions(collection=points, properties=["label"], scale=10,
                           tileScale=4, geometries=True)
            .filter(ee.Filter.notNull(BANDS + ["label"])))


def build_eval_table():
    reference = reference_image()
    gaul = districts_fc()
    tables = []
    for name in DISTRICTS:
        region = gaul.filter(ee.Filter.eq("ADM2_NAME", name)).geometry()
        print(f"  sampling {name} ...", flush=True)
        stack = _composite(region, clip=True).select(BANDS).addBands(reference)
        # Lazy: nothing is computed until the export runs, so no retry is needed here.
        # Seed is derived from the name rather than PYTHONHASHSEED-dependent hash().
        samples = stack.stratifiedSample(
            numPoints=POINTS_PER_CLASS, classBand="reference", region=region,
            scale=10, seed=sum(map(ord, name)), tileScale=8, geometries=True)
        tables.append(samples.map(lambda f: f.set("district", name)))
    merged = tables[0]
    for t in tables[1:]:
        merged = merged.merge(t)
    return merged.filter(ee.Filter.notNull(BANDS + ["reference"]))


def cmd_sample(_args):
    init_ee()
    print("checking training label integrity ...")
    check_labels()
    print("building training table (now including sand_points_mp) ...")
    train = build_training_table()
    print("building per-district evaluation table ...")
    evaluation = build_eval_table()
    wait_for([
        export_to_asset(train, TRAIN_ASSET, "district_train_table"),
        export_to_asset(evaluation, EVAL_ASSET, "district_eval_table"),
    ])


def make_classifier(model):
    """Exactly the hyperparameters the application ships, so this measures the product."""
    if model == "rf":
        return ee.Classifier.smileRandomForest(
            numberOfTrees=200, minLeafPopulation=1, bagFraction=0.3, maxNodes=10)
    if model == "svm":
        return ee.Classifier.libsvm(kernelType="RBF", gamma=1.0, cost=100.0)
    if model in ("gtb", "xgb"):
        return ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=100, shrinkage=0.1, maxNodes=10)
    if model == "cart":
        return ee.Classifier.smileCart(maxNodes=20)
    if model == "knn":
        return ee.Classifier.smileKNN(k=5)
    raise ValueError(model)


def _metrics(classified, district):
    subset = classified.filter(ee.Filter.eq("district", district))
    matrix = subset.errorMatrix("reference", "classification", CLASS_IDS)
    return ee.Dictionary({
        "n": subset.size(),
        "overall": matrix.accuracy(),
        "kappa": matrix.kappa(),
        "recall": matrix.producersAccuracy().toList().flatten(),
        "precision": matrix.consumersAccuracy().toList().flatten(),
        "matrix": matrix.array().toList(),
    })


def cmd_run(_args):
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    eval_fc = ee.FeatureCollection(EVAL_ASSET)
    n_train = retry(lambda: train_fc.size().getInfo())
    n_eval = retry(lambda: eval_fc.size().getInfo())
    print(f"train {n_train} points   eval {n_eval} points   "
          f"{len(CLASS_IDS)} classes", flush=True)

    results = {}
    for model in MODELS:
        classifier = make_classifier(model).train(
            features=train_fc, classProperty="label", inputProperties=BANDS)
        classified = eval_fc.classify(classifier)
        started = time.time()
        # One request per model rather than per district-model pair: 10 small
        # matrices cost far less as a single computed list than as 10 round trips
        # against a throttled quota.
        per_district = retry(lambda: ee.List(
            [_metrics(classified, d) for d in DISTRICTS]).getInfo())
        results[model] = dict(zip(DISTRICTS, per_district))
        best = max(DISTRICTS, key=lambda d: results[model][d]["overall"])
        mean = sum(results[model][d]["overall"] for d in DISTRICTS) / len(DISTRICTS)
        print(f"{model:5s} mean {mean * 100:5.2f}%   best {best} "
              f"{results[model][best]['overall'] * 100:5.2f}%   "
              f"({time.time() - started:.0f}s)", flush=True)
        _save(RESULTS_PATH, {"class_names": CLASS_NAMES, "class_ids": CLASS_IDS,
                             "districts": DISTRICTS, "models": MODELS,
                             "n_train": n_train, "n_eval": n_eval,
                             "results": results})
    print(f"wrote {RESULTS_PATH}")


def cmd_sites(_args):
    """Descriptive classification of the two Rajasthan sites. No proxy scoring.

    Resumable: each (site, model) pair is saved as soon as it lands and skipped on
    re-entry. Classifying 5,300 km2 of Thar five times over a throttled quota is a
    long job, and losing all of it to one interruption is avoidable.
    """
    init_ee()
    train_fc = ee.FeatureCollection(TRAIN_ASSET)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    out = {}
    if os.path.exists(SITES_PATH):
        with open(SITES_PATH) as fh:
            out = json.load(fh).get("sites", {})
    for site, box in SITES.items():
        region = ee.Geometry.Rectangle(box)
        scale = SITE_SCALE[site]
        composite = _composite(region, clip=True).select(BANDS)
        if site not in out:
            wc_area = retry(lambda: ee.Image.cat([
                worldcover.eq(60).rename("bare"), worldcover.eq(50).rename("built"),
            ]).multiply(ee.Image.pixelArea()).divide(1e4).reduceRegion(
                ee.Reducer.sum(), region, scale=scale, maxPixels=1e10).getInfo())
            out[site] = {"worldcover_bare_ha": wc_area["bare"],
                         "worldcover_built_ha": wc_area["built"], "models": {}}
            _save(SITES_PATH, {"arora_sand_ha": ARORA_SAND_HA, "sites": out})
        for model in MODELS:
            if model in out[site]["models"]:
                print(f"{site[:28]:30s} {model:5s} cached", flush=True)
                continue
            classifier = make_classifier(model).train(
                features=train_fc, classProperty="label", inputProperties=BANDS)
            classified = composite.classify(classifier)
            hist = retry(lambda: classified.reduceRegion(
                ee.Reducer.frequencyHistogram(), region, scale=scale,
                maxPixels=1e10).getInfo())["classification"] or {}
            total = sum(float(v) for v in hist.values()) or 1.0
            if not hist:
                raise SystemExit(f"{site}/{model}: empty histogram, refusing to "
                                 f"report a class distribution built from nothing")
            pct = {LAND_COVER_CLASSES[i]["name"]:
                   100.0 * float(hist.get(str(i), 0)) / total for i in CLASS_IDS}
            out[site]["models"][model] = pct
            print(f"{site[:28]:30s} {model:5s} " + "  ".join(
                f"{k} {v:5.1f}%" for k, v in pct.items()), flush=True)
            _save(SITES_PATH, {"arora_sand_ha": ARORA_SAND_HA, "sites": out})
    print(f"wrote {SITES_PATH}")


def _save(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").set_defaults(func=cmd_sample)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("sites").set_defaults(func=cmd_sites)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
