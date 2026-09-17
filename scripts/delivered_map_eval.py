#!/usr/bin/env python3
"""Agreement of the raster the dashboard actually delivers, not the raw one.

Every accuracy in the paper is computed on the unfiltered per-pixel
classification. The dashboard serves a 3x3 majority-filtered raster, and the
paper's own limitation section says that product carries no measured agreement
figure at all. This measures it, on the same frozen test reference sample, so
the delivered map is assessed rather than assumed equivalent.

Two quantities come out of it: the agreement of the filtered raster against the
same reference points, per class as well as overall, and what the filter does to
the class areas of a district -- the water class in particular, since water
bodies are the smallest connected features in the inventory and a majority
filter erodes exactly those.

Writes doc/assets/delivered_map_eval.json. Needs Earth Engine.
"""
import argparse
import csv
import json
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
from evaluation.references import REFERENCE_LABELS, madhya_pradesh_districts  # noqa: E402

ARCHIVE = REPO / "doc" / "assets" / "result_archive"
OUTPUT = REPO / "doc" / "assets" / "delivered_map_eval_{season}.json"
NAME_TO_LABEL = {name: value for value, name in REFERENCE_LABELS.items()}


def with_retry(label, call, attempts=6):
    """Earth Engine's Restricted Mode rejects concurrent work; wait and repeat."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ee.EEException as error:
            if attempt == attempts:
                raise
            print(f"  retry {label} ({attempt}/{attempts}): {error}", flush=True)
            time.sleep(20 * attempt)


def reference_points(season, districts, role="test"):
    features = []
    with (ARCHIVE / "predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["season"] != season or row["role"] != role:
                continue
            if districts and row["district"] not in districts:
                continue
            features.append(ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"reference": NAME_TO_LABEL[row["reference_label"]]}))
    return ee.FeatureCollection(features), len(features)


def classified_pair(region, season, classifier):
    """The raw classification and the delivered 3x3 majority-filtered one."""
    dates = config.SEASONS[season]
    collection = _build_collection(region, dates["start"], dates["end"], 15)
    composite = _add_spectral_indices(collection.median())
    composite = _add_sar(composite, region, dates["start"], dates["end"])
    raw = composite.select(config.BANDS).classify(classifier).rename("raw")
    delivered = raw.focalMode(radius=1, kernelType="square", units="pixels") \
        .rename("delivered")
    return raw.addBands(delivered)


def confusion(rows, column):
    matrix = Counter()
    for row in rows:
        matrix[(row["reference"], row[column])] += 1
    return matrix


def scores(matrix):
    total = sum(matrix.values())
    correct = sum(count for (a, b), count in matrix.items() if a == b)
    per_class = {}
    for value, name in REFERENCE_LABELS.items():
        support = sum(c for (a, _), c in matrix.items() if a == value)
        hit = matrix.get((value, value), 0)
        per_class[name] = {
            "support": support,
            "recall_pct": round(hit / support * 100, 2) if support else None,
        }
    return {"n": total,
            "overall_agreement_pct": round(correct / total * 100, 2),
            "per_class": per_class}


def class_areas(image, band, region, scale):
    areas = ee.Image.pixelArea().addBands(image.select(band)).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region, scale=scale, maxPixels=1e13, tileScale=8).getInfo()
    return {REFERENCE_LABELS[int(group["class"])]: round(group["sum"] / 1e6, 4)
            for group in areas["groups"] if int(group["class"]) in REFERENCE_LABELS}


def run(args):
    init_ee()
    dates = config.SEASONS[args.season]
    classifier = make_classifier("gtb").train(
        features=sample_training_points(
            merge_feature_collections(config.FEATURE_COLLECTIONS.values()),
            start_date=dates["start"], end_date=dates["end"]),
        classProperty="label", inputProperties=config.BANDS)

    districts = madhya_pradesh_districts()
    names = sorted({row["district"] for row in
                    csv.DictReader((ARCHIVE / "predictions.csv").open())
                    if row["season"] == args.season and row["role"] == "test"})
    if args.districts:
        names = [name for name in names if name in args.districts]

    rows = []
    for index, name in enumerate(names, start=1):
        region = ee.Feature(districts.filter(
            ee.Filter.eq("ADM2_NAME", name)).first()).geometry()
        points, count = reference_points(args.season, {name})
        sampled = with_retry(name, lambda: classified_pair(
            region, args.season, classifier).sampleRegions(
                collection=points, properties=["reference"], scale=10,
                tileScale=4).getInfo())["features"]
        rows.extend(row["properties"] for row in sampled)
        print(f"  [{index}/{len(names)}] {name}: {len(sampled)}/{count} points "
              f"sampled, running n={len(rows)}", flush=True)

    kept = [row for row in rows if "raw" in row and "delivered" in row]
    report = {
        "season": args.season,
        "districts": len(names),
        "points_sampled": len(rows),
        "points_with_both_rasters": len(kept),
        "raw": scores(confusion(kept, "raw")),
        "delivered": scores(confusion(kept, "delivered")),
        "points_changed_by_the_filter": sum(
            row["raw"] != row["delivered"] for row in kept),
    }
    report["delta_pp"] = round(
        report["delivered"]["overall_agreement_pct"]
        - report["raw"]["overall_agreement_pct"], 2)

    # Written before the class-area pass: that pass is a single expensive
    # reduction and losing forty minutes of sampling to it would be silly.
    output = Path(str(OUTPUT).format(season=args.season))
    output.write_text(json.dumps(report, indent=1) + "\n")

    if args.area_district:
        region = ee.Feature(districts.filter(
            ee.Filter.eq("ADM2_NAME", args.area_district)).first()).geometry()
        image = classified_pair(region, args.season, classifier).clip(region)
        scale = args.area_scale
        try:
            areas = {"raw": with_retry("raw areas", lambda: class_areas(
                         image, "raw", region, scale)),
                     "delivered": with_retry("filtered areas", lambda: class_areas(
                         image, "delivered", region, scale))}
        except ee.EEException as error:
            print(f"  class areas at {scale} m failed ({error}); "
                  f"falling back to 30 m", flush=True)
            scale = 30
            areas = {"raw": class_areas(image, "raw", region, scale),
                     "delivered": class_areas(image, "delivered", region, scale)}
        report["class_areas_km2"] = {
            "district": args.area_district, "scale_m": scale, **areas}
        output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nraw       {report['raw']['overall_agreement_pct']}%")
    print(f"delivered {report['delivered']['overall_agreement_pct']}%  "
          f"({report['delta_pp']:+} pp, "
          f"{report['points_changed_by_the_filter']} of {len(kept)} points "
          f"relabelled by the filter)")
    for name in REFERENCE_LABELS.values():
        raw = report["raw"]["per_class"][name]
        delivered = report["delivered"]["per_class"][name]
        print(f"  {name:12} recall {raw['recall_pct']:6.2f} -> "
              f"{delivered['recall_pct']:6.2f}  (n={raw['support']})")
    if args.area_district:
        areas = report["class_areas_km2"]
        print(f"\nclass areas in {areas['district']} at {areas['scale_m']} m:")
        for name in REFERENCE_LABELS.values():
            before = areas["raw"].get(name, 0.0)
            after = areas["delivered"].get(name, 0.0)
            change = (after - before) / before * 100 if before else float("nan")
            print(f"  {name:12} {before:10.2f} -> {after:10.2f} km2 "
                  f"({change:+.1f}%)")
    print(f"\nwrote {output.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter", choices=sorted(config.SEASONS))
    parser.add_argument("--districts", nargs="*", default=None)
    parser.add_argument("--area-district", default="Jabalpur")
    parser.add_argument("--area-scale", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
