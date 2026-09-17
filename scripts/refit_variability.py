#!/usr/bin/env python3
"""Repeated fits on one frozen table: how much of the rerun difference is fitting.

The reproduction check compared two runs that differed in everything at once --
imagery request, consensus construction, stratified sample, feature sampling and
model fit -- and attributed the difference to the classifier by elimination. That
is not a measurement of fitting variability.

This is. The training table and the evaluation table are each sampled once,
pulled client-side, and rebuilt as constant feature collections, so every repeat
sees identical rows, identical feature values and identical column order. The
only thing that varies between repeats is the fit. Each repeat classifies the
same frozen evaluation rows, so the disagreement reported here is a count of
changed point predictions and not a confusion-matrix redistribution statistic.

Writes doc/assets/refit_variability.json. Needs Earth Engine.
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend import config  # noqa: E402
from backend.gee_classifier import (  # noqa: E402
    _add_sar,
    _add_spectral_indices,
    _build_collection,
    init_ee,
    make_classifier,
    merge_feature_collections,
    sample_training_points,
)
from evaluation.references import REFERENCE_LABELS  # noqa: E402

ARCHIVE = REPO / "doc" / "assets" / "result_archive"
OUTPUT = REPO / "doc" / "assets" / "refit_variability.json"
LABEL_TO_NAME = dict(REFERENCE_LABELS)


def with_retry(label, call, attempts=8):
    """Earth Engine's Restricted Mode rejects concurrent work; wait and repeat."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ee.EEException as error:
            if attempt == attempts:
                raise
            print(f"  retry {label} ({attempt}/{attempts}): {error}", flush=True)
            time.sleep(20 * attempt)


def pull(collection, properties, label="sample"):
    """Pull a sampled collection client-side as plain dictionaries."""
    rows = with_retry(label, lambda: collection.getInfo())["features"]
    return [{key: row["properties"][key] for key in properties} for row in rows]


def constant_collection(rows):
    """Rebuild pulled rows as a constant collection.

    Everything downstream then reads the same numbers in the same order however
    often it is re-evaluated, which is what makes a repeat a repeat.
    """
    return ee.FeatureCollection([ee.Feature(None, row) for row in rows])


def evaluation_points(season, role="test", limit=None):
    """Frozen reference coordinates and labels per district, from the archive."""
    name_to_label = {name: value for value, name in REFERENCE_LABELS.items()}
    by_district, total = {}, 0
    with (ARCHIVE / "predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["season"] != season or row["role"] != role:
                continue
            by_district.setdefault(row["district"], []).append(ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"reference": name_to_label[row["reference_label"]],
                 "district": row["district"]}))
            total += 1
            if limit and total >= limit:
                break
    return by_district


def pull_evaluation(season, bands, limit=None):
    """Sample the feature stack district by district and pull the rows.

    One request over the bounding box of all 29 test districts asks Earth Engine
    to build a statewide composite before it samples anything, which is the
    slowest possible way to read 2,900 pixels. A district is small enough that
    each request finishes, and the rows are identical either way.
    """
    dates = config.SEASONS[season]
    rows = []
    districts = evaluation_points(season, limit=limit)
    for index, (name, features) in enumerate(sorted(districts.items()), start=1):
        points = ee.FeatureCollection(features)
        region = points.geometry().bounds()
        collection = _build_collection(region, dates["start"], dates["end"], 15)
        composite = _add_spectral_indices(collection.median())
        composite = _add_sar(composite, region, dates["start"], dates["end"])
        sampled = composite.select(bands).sampleRegions(
            collection=points, properties=["reference", "district"], scale=10,
            tileScale=4).filter(ee.Filter.notNull(bands + ["reference"]))
        rows.extend(pull(sampled, bands + ["reference", "district"], name))
        print(f"  [{index}/{len(districts)}] {name}: {len(rows)} rows",
              flush=True)
    return rows


def repeat_fits(model, training_for, evaluation, bands, repeats):
    """Classify the frozen evaluation rows `repeats` times and report the spread.

    `training_for(index)` returns the training collection for one repeat, which
    is what separates the three modes: identical every time, the same rows in a
    different order, or freshly sampled from Earth Engine.
    """
    runs = []
    for index in range(repeats):
        classifier = make_classifier(model).train(
            features=training_for(index), classProperty="label",
            inputProperties=bands)
        rows = with_retry(f"{model} fit {index + 1}",
                          lambda: evaluation.classify(classifier).getInfo()
                          )["features"]
        runs.append([(row["properties"]["reference"],
                      row["properties"]["classification"],
                      row["properties"]["district"]) for row in rows])
        accuracy = sum(a == b for a, b, _ in runs[-1]) / len(runs[-1])
        print(f"  {model} fit {index + 1}/{repeats}: agreement "
              f"{accuracy * 100:.2f}% on {len(runs[-1])} frozen rows", flush=True)
    return runs


def summarise(model, mode, runs):
    accuracies = [sum(a == b for a, b, _ in run) / len(run) for run in runs]
    first = runs[0]
    changed_vs_first, changed_districts = [], Counter()
    for run in runs[1:]:
        changed = [i for i in range(len(first)) if first[i][1] != run[i][1]]
        changed_vs_first.append(len(changed))
        for i in changed:
            changed_districts[first[i][2]] += 1
    ever_changed = {i for run in runs[1:] for i in range(len(first))
                    if first[i][1] != run[i][1]}
    return {
        "model": model,
        "mode": mode,
        "repeats": len(runs),
        "rows": len(first),
        "agreement_pct": [round(a * 100, 3) for a in accuracies],
        "agreement_range_pp": round((max(accuracies) - min(accuracies)) * 100, 3),
        "changed_point_predictions_vs_first_fit": changed_vs_first,
        "points_changed_by_any_repeat": len(ever_changed),
        "districts_touched": len(changed_districts),
    }


# Three modes, in increasing order of what is allowed to vary between repeats.
# Together they say where a rerun difference is produced, which the paper's
# reproduction check could only guess at.
MODES = ("frozen", "shuffled", "resampled")
MODE_NOTE = {
    "frozen": "identical training rows, values and order",
    "shuffled": "identical training rows and values, order permuted per repeat",
    "resampled": "training table re-sampled from Earth Engine per repeat",
}


def run(args):
    init_ee()
    bands = list(config.BANDS)
    dates = config.SEASONS[args.season]

    def fresh_training():
        return sample_training_points(
            merge_feature_collections(config.FEATURE_COLLECTIONS.values()),
            start_date=dates["start"], end_date=dates["end"])

    print(f"freezing the training table ({args.season})", flush=True)
    training_rows = pull(fresh_training(), bands + ["label"], "training table")
    frozen_training = constant_collection(training_rows)
    print(f"  {len(training_rows)} training rows frozen", flush=True)

    print("freezing the evaluation table", flush=True)
    evaluation_features = pull_evaluation(args.season, bands, args.points)
    evaluation = constant_collection(evaluation_features)
    print(f"  {len(evaluation_features)} evaluation rows frozen", flush=True)

    def training_for(mode):
        if mode == "frozen":
            return lambda index: frozen_training
        if mode == "shuffled":
            def shuffled(index):
                order = list(range(len(training_rows)))
                random.Random(args.shuffle_seed + index).shuffle(order)
                return constant_collection([training_rows[i] for i in order])
            return shuffled
        return lambda index: fresh_training()

    report = {
        "season": args.season,
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_features),
        "repeats": args.repeats,
        "mode_notes": MODE_NOTE,
        "runs": [],
    }
    for mode in args.modes:
        for model in args.models:
            print(f"\nrepeated fits, {model}, mode={mode} "
                  f"({MODE_NOTE[mode]}):", flush=True)
            report["runs"].append(summarise(model, mode, repeat_fits(
                model, training_for(mode), evaluation, bands, args.repeats)))
            # Written after every model, not once at the end: this run takes an
            # hour and a dropped connection at minute 59 used to lose all of it.
            OUTPUT.write_text(json.dumps(report, indent=1) + "\n")

    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {OUTPUT.relative_to(REPO)}")
    for entry in report["runs"]:
        print(f"  {entry['model']:4} {entry['mode']:10} agreement "
              f"{min(entry['agreement_pct']):.2f}-{max(entry['agreement_pct']):.2f}%"
              f" (range {entry['agreement_range_pp']:.2f} pp), "
              f"{entry['points_changed_by_any_repeat']} of {entry['rows']} "
              f"point predictions changed in at least one repeat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter", choices=sorted(config.SEASONS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--points", type=int, default=None,
                        help="cap on frozen evaluation rows; default is all")
    parser.add_argument("--models", nargs="+", default=["gtb", "rf"])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--shuffle-seed", type=int, default=20260909)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
