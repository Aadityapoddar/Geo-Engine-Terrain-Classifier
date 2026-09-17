#!/usr/bin/env python3
"""Export the row-level prediction table the result archive was missing.

Every metric in this paper is derived from per-district confusion matrices,
which are the sufficient statistic for all of them. They are not, however, an
auditable record: a reader cannot recompute a stratum weight, re-stratify the
sample, check a buffer, or find the pixel behind a disputed label from a matrix.
The evaluation path built the classified sample server-side and brought back
only ee.ConfusionMatrix, so those rows existed in Earth Engine and were never
written down.

This writes them down. It rebuilds the same stratified reference sample the
evaluation used --- same reference construction, same seed, same 20-per-class
quota, same 100 m training buffer --- classifies it with the frozen production
model, and pulls back one row per sampled pixel:

    sample_id, run_id, district, role, season, lon, lat,
    reference_label, predicted_label, stratum_id, agreement

Every model is classified on the same rows in the same pass, so the classifier
comparison is a comparison of columns in one table rather than of numbers from
separate runs.

The reference sample is reproducible: stratifiedSample is seeded and the
consensus is a pure function of the season window, and in practice the exported
rows recover the stored per-district reference totals exactly. The predictions
do not always recover the stored ones to the last pixel --- a spot check found
two districts agreeing exactly and one differing on 2 of 100 points --- because
neither the Sentinel archives nor Earth Engine's server-side classifiers
guarantee bit-identical results across months. That is the reason this table
exists rather than an argument against it: with the rows written down, the
paper's numbers are recomputed from a record that can be inspected, instead of
being carried forward from matrices no one can re-derive. Each district still
reports whether it reproduced its stored matrix, so the size of that drift is
visible.

Checkpointed per district, so a run can be interrupted and resumed, and
shardable with --worker/--workers so several can run at once.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ee  # noqa: E402

from backend.config import BAND_STACKS, BANDS, FULL_BANDS, SEASONS  # noqa: E402
from backend.gee_classifier import (  # noqa: E402
    build_sentinel_composite,
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.assets import training_points  # noqa: E402
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    exclude_training_neighbours,
    madhya_pradesh_districts,
)
from evaluation.splits import district_splits  # noqa: E402
from scripts.run_seasonal_evaluation import BLOCK_SEED  # noqa: E402

ASSETS = REPO / "doc" / "assets"
ARCHIVE = ASSETS / "result_archive"
CHECKPOINT = ASSETS / "predictions_checkpoint.json"
OUT = ARCHIVE / "predictions.csv"
SHARD_GLOB = "mp_external_shards_v7_worker_*.json"

CONDITION = "after"
# The five headline classifiers on the production stack, plus the leading model
# on each alternative stack, so the feature-stack table of the paper comes from
# the same rows as the classifier table.
HEADLINE_MODELS = ("gtb", "rf", "svm", "knn", "cart")
STACK_VARIANTS = ("b19", "b22", "b27")
LEADER = "gtb"
LABELS = list(REFERENCE_LABELS.values())
BASE_FIELDS = ["sample_id", "season", "district", "role", "lon", "lat",
               "reference_label", "stratum_id"]


def load_checkpoint(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_checkpoint(path, state):
    path.write_text(json.dumps(state) + "\n")


def stored_matrices():
    """The per-district matrices this export is compared against."""
    matrices = {}
    for path in sorted(ASSETS.glob(SHARD_GLOB)):
        for key, shard in json.loads(path.read_text())["shards"].items():
            matrices[key] = shard["matrices"]
    return matrices


def model_keys():
    keys = [f"{CONDITION}-{model}" for model in HEADLINE_MODELS]
    keys += [f"{CONDITION}-{LEADER}-{stack}" for stack in STACK_VARIANTS
             if list(BAND_STACKS[stack]) != list(BANDS)]
    return keys


def trained_classifiers(season):
    """Every classifier this export scores, all fitted on one sampled table."""
    dates = SEASONS[season]
    table = sample_training_points(
        training_points(CONDITION),
        start_date=dates["start"], end_date=dates["end"], bands=FULL_BANDS)
    classifiers = {}
    for model in HEADLINE_MODELS:
        classifiers[f"{CONDITION}-{model}"] = make_classifier(model).train(
            features=table, classProperty="label", inputProperties=BANDS)
    for stack in STACK_VARIANTS:
        bands = list(BAND_STACKS[stack])
        if bands == list(BANDS):
            continue
        classifiers[f"{CONDITION}-{LEADER}-{stack}"] = \
            make_classifier(LEADER).train(
                features=table, classProperty="label", inputProperties=bands)
    return classifiers


def with_retry(label, call, attempts=8):
    """Earth Engine hands back a 429 under Restricted Mode; wait and retry.

    The concurrency limit is per project, not per request, so a failure here
    means somebody (often another shard of this same script) is mid-request.
    Backing off and retrying is the whole fix; nothing about the computation
    needs to change.
    """
    delay = 20
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ee.ee_exception.EEException as error:
            if "Too Many Requests" not in str(error) and \
                    "concurrency" not in str(error):
                raise
            print(f"  {label}: rate limited, retry {attempt} in {delay}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError(f"{label}: still rate limited after {attempts} attempts")


def district_rows(district, name, season, leakage, classifiers):
    """The same sample the evaluation drew, classified and brought back."""
    dates = SEASONS[season]
    region = district.geometry()
    reference = build_reference_image(region, season, dates["start"], dates["end"])
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
        region, start_date=dates["start"], end_date=dates["end"])
    table = (
        composite.select(FULL_BANDS)
        .sampleRegions(collection=samples,
                       properties=["reference", "district", "season"],
                       scale=10, tileScale=8, geometries=True)
        .filter(ee.Filter.notNull(FULL_BANDS + ["reference", "district"]))
    )
    # One classify() per model on the same table, renamed so all of them come
    # back in a single round trip and are guaranteed to describe the same rows.
    for key, classifier in classifiers.items():
        table = table.classify(classifier, key)
    columns = ["reference"] + list(classifiers)
    payload = with_retry(
        f"{season}:{name}",
        lambda: table.select(columns, retainGeometry=True).getInfo())

    rows = []
    for index, feature in enumerate(payload["features"]):
        lon, lat = feature["geometry"]["coordinates"][:2]
        properties = feature["properties"]
        reference_index = int(properties["reference"])
        row = {
            "sample_id": f"{season}:{name}:{index:04d}",
            "season": season,
            "district": name,
            "role": None,  # filled in by the caller, which knows the split
            "lon": round(lon, 7),
            "lat": round(lat, 7),
            "reference_label": LABELS[reference_index],
            "stratum_id": f"{season}:{name}:{LABELS[reference_index]}",
        }
        for key in classifiers:
            row[f"predicted_{key}"] = LABELS[int(properties[key])]
        rows.append(row)
    return rows


def matrix_from_rows(rows, key):
    size = len(LABELS)
    matrix = [[0] * size for _ in range(size)]
    for row in rows:
        matrix[LABELS.index(row["reference_label"])][
            LABELS.index(row[f"predicted_{key}"])] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=list(SEASONS))
    parser.add_argument("--districts", nargs="*", default=None,
                        help="default: every district in the split manifest")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--merge-only", action="store_true",
                        help="write the CSV from existing checkpoints, no Earth Engine")
    args = parser.parse_args()

    splits = district_splits()
    role = {name: "test" for name in splits["test"]}
    role.update({name: "development" for name in splits["development"]})
    role.update({name: "training-withheld" for name in splits["training"]})
    wanted = args.districts or sorted(role)
    keys = model_keys()
    fields = BASE_FIELDS + [f"predicted_{key}" for key in keys]

    if args.merge_only:
        # Each worker keeps its own checkpoint so that concurrent writers never
        # clobber one another; merging is a separate, offline step.
        state = {}
        for path in sorted(args.checkpoint.parent.glob(
                args.checkpoint.name.replace(".json", "*.json"))):
            state.update(json.loads(path.read_text()))
        write_csv(args.output, fields, state)
        return

    init_ee()
    collection = madhya_pradesh_districts()
    leakage = training_points("before").merge(training_points(CONDITION))
    stored = stored_matrices()
    state = load_checkpoint(args.checkpoint)

    for season in args.seasons:
        classifiers = trained_classifiers(season)
        for index, name in enumerate(wanted):
            if index % args.workers != args.worker:
                continue
            key = f"{season}:{name}"
            if key in state:
                continue
            start = time.time()
            district = ee.Feature(
                collection.filter(ee.Filter.eq("ADM2_NAME", name)).first())
            rows = district_rows(district, name, season, leakage, classifiers)
            for row in rows:
                row["role"] = role.get(name, "unassigned")
            rebuilt = {model: matrix_from_rows(rows, model) for model in keys}
            expected = stored.get(key, {})
            same = [model for model in keys
                    if expected.get(model) == rebuilt[model]]
            drift = sum(
                abs(a - b)
                for model in keys if model in expected
                for row_new, row_old in zip(rebuilt[model], expected[model])
                for a, b in zip(row_new, row_old)) // 2
            state[key] = {"rows": rows,
                          "models_reproducing_stored_matrix": same,
                          "points_moved_vs_stored": drift,
                          "rebuilt_matrices": rebuilt}
            save_checkpoint(args.checkpoint, state)
            print(f"{key:26s} n={len(rows):4d}  "
                  f"{len(same)}/{len(keys)} models reproduce stored, "
                  f"{drift} point-moves  {time.time() - start:5.0f}s", flush=True)

    write_csv(args.output, fields, state)


def write_csv(path, fields, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(state):
            writer.writerows(state[key]["rows"])
    total = sum(len(state[key]["rows"]) for key in state)
    moved = sum(state[key].get("points_moved_vs_stored", 0) for key in state)
    print(f"\nwrote {path} ({total} rows over {len(state)} district-seasons; "
          f"{moved} point-level differences against the stored matrices)")


if __name__ == "__main__":
    main()
