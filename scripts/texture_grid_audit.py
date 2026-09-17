#!/usr/bin/env python3
"""What grid the GLCM texture bands were actually computed on.

`glcmTexture(size=3)` is a neighbourhood operation, so its footprint in metres
is whatever the pixel grid underneath it is. The paper described that footprint
as 7x7 pixels on a 10 m grid without ever recording the projection the composite
carries, and an Earth Engine composite of many scenes carries the default WGS84
1-degree projection rather than the 10 m grid of its inputs. When an image has a
default projection, the neighbourhood operation is evaluated in the projection of
the *request*, so the texture footprint is a property of the call site and not of
the feature code.

This script records that explicitly:

  1. the CRS, transform and nominal scale of every intermediate image, from a
     single scene through to the classified output;
  2. the sampled texture values at a range of request scales, which is the test
     of whether the footprint follows the request;
  3. the same pipeline with the grey-level image pinned to an explicit 10 m grid
     with .reproject, which is what the paper's description assumed; and
  4. how many predictions change between those variants, which is what any of it
     costs.

Writes doc/assets/texture_grid_audit.json. Needs Earth Engine.
"""
import argparse
import csv
import json
import sys
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

ARCHIVE = REPO / "doc" / "assets" / "result_archive"
OUTPUT = REPO / "doc" / "assets" / "texture_grid_audit.json"
TEXTURE_BANDS = ["g_contrast", "g_var", "g_idm", "s_contrast", "s_var", "s_ent"]
REQUEST_SCALES = (10, 20, 30, 60, 100)


def audit_points(district, season, limit):
    """Reference sample coordinates for one district, from the frozen archive."""
    features = []
    with (ARCHIVE / "predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["season"] != season or row["district"] != district:
                continue
            features.append(ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"sample_id": row["sample_id"]}))
            if len(features) >= limit:
                break
    return ee.FeatureCollection(features)


def stack(region, season, pin_scale=None):
    """The production feature stack over `region`, optionally on a pinned grid.

    `pin_scale` reproduces what the manuscript's description assumed: the
    grey-level images that feed glcmTexture pinned to an explicit grid, so the
    neighbourhood footprint stops depending on the request.
    """
    dates = config.SEASONS[season]
    collection = _build_collection(region, dates["start"], dates["end"], 15)
    composite = collection.median()
    if pin_scale is not None:
        composite = composite.reproject(crs="EPSG:4326", scale=pin_scale)
    composite = _add_spectral_indices(composite)
    return _add_sar(composite, region, dates["start"], dates["end"])


def projections(region, season):
    """CRS, transform and nominal scale at every stage of the stack."""
    dates = config.SEASONS[season]
    collection = _build_collection(region, dates["start"], dates["end"], 15)
    scene = ee.Image(collection.first())
    composite = collection.median()
    with_indices = _add_spectral_indices(composite)
    full = _add_sar(with_indices, region, dates["start"], dates["end"])
    stages = {
        "single S2 scene, B8": scene.select("B8"),
        "single S2 scene, B11": scene.select("B11"),
        "cloud-masked median, B8": composite.select("B8"),
        "composite with indices, NDVI": with_indices.select("NDVI"),
        "optical GLCM band, g_contrast": with_indices.select("g_contrast"),
        "SAR GLCM band, s_contrast": full.select("s_contrast"),
    }
    out = {}
    for name, image in stages.items():
        projection = image.projection()
        info = ee.Dictionary({
            "crs": projection.crs(),
            "transform": projection.transform(),
            "nominal_scale_m": projection.nominalScale(),
        }).getInfo()
        out[name] = info
        print(f"  {name:34} {info['crs']:12} "
              f"nominal scale {info['nominal_scale_m']:.3f} m")
    return out


def sampled(image, points, scale, bands):
    rows = image.select(bands).sampleRegions(
        collection=points, scale=scale, tileScale=4,
        properties=["sample_id"]).getInfo()["features"]
    return {row["properties"]["sample_id"]: row["properties"] for row in rows}


def compare(reference, other, bands):
    """Mean absolute difference per band over the points both variants kept."""
    shared = sorted(set(reference) & set(other))
    out = {}
    for band in bands:
        deltas = [abs(reference[k][band] - other[k][band]) for k in shared
                  if reference[k].get(band) is not None
                  and other[k].get(band) is not None]
        out[band] = round(sum(deltas) / len(deltas), 6) if deltas else None
    out["n"] = len(shared)
    return out


def delivered_render_scales():
    """The request scale the dashboard's own renderer asks for, by AOI width.

    backend.gee_classifier._static_overlay asks for width_m / 10 pixels, capped
    at 2048, so an AOI wider than about 20 km is rendered coarser than 10 m and
    its texture bands are therefore computed on a coarser grid than the one the
    classifier was trained and scored on.
    """
    out = {}
    for width_km in (5, 10, 20, 50, 100, 200):
        width_m = width_km * 1000
        px = max(256, min(2048, int(width_m / 10)))
        scale = width_m / px
        out[width_km] = {
            "render_pixels": px,
            "request_scale_m": round(scale, 2),
            "glcm_footprint_m": round(scale * 7, 1),
        }
        print(f"  AOI {width_km:4} km wide -> {px:5} px, request scale "
              f"{scale:7.2f} m, 7x7 GLCM footprint {scale * 7:8.1f} m")
    return out


def run(args):
    init_ee()
    district = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.eq("ADM2_NAME", args.district)).first()
    region = ee.Feature(district).geometry()
    points = audit_points(args.district, args.season, args.points)

    print(f"projection of every stage ({args.season}, {args.district}):")
    report = {
        "district": args.district,
        "season": args.season,
        "points": args.points,
        "stage_projections": projections(region, args.season),
    }

    default_stack = stack(region, args.season)
    classifier = make_classifier("gtb").train(
        features=sample_training_points(
            merge_feature_collections(config.FEATURE_COLLECTIONS.values()),
            start_date=config.SEASONS[args.season]["start"],
            end_date=config.SEASONS[args.season]["end"]),
        classProperty="label", inputProperties=config.BANDS)

    def predictions(image, scale):
        classified = image.select(config.BANDS).classify(classifier)
        rows = classified.sampleRegions(
            collection=points, scale=scale, tileScale=4,
            properties=["sample_id"]).getInfo()["features"]
        return {row["properties"]["sample_id"]: row["properties"]["classification"]
                for row in rows}

    print("\ntexture values and predictions by request scale:")
    baseline = sampled(default_stack, points, 10, TEXTURE_BANDS)
    baseline_prediction = predictions(default_stack, 10)
    report["request_scales"] = {}
    for scale in REQUEST_SCALES:
        values = baseline if scale == 10 else sampled(
            default_stack, points, scale, TEXTURE_BANDS)
        predicted = baseline_prediction if scale == 10 else predictions(
            default_stack, scale)
        shared = sorted(set(baseline_prediction) & set(predicted))
        changed = sum(baseline_prediction[k] != predicted[k] for k in shared)
        entry = compare(baseline, values, TEXTURE_BANDS)
        entry["predictions_changed_vs_scale_10"] = changed
        entry["predictions_compared"] = len(shared)
        report["request_scales"][scale] = entry
        print(f"  scale {scale:4} m: mean |delta| g_contrast "
              f"{entry['g_contrast']}, s_contrast {entry['s_contrast']}, "
              f"predictions changed {changed}/{len(shared)}")

    print("\ngrey-level image pinned with .reproject, sampled at 10 m:")
    report["pinned_grid"] = {}
    for pin in (10, 30):
        pinned = stack(region, args.season, pin_scale=pin)
        values = sampled(pinned, points, 10, TEXTURE_BANDS)
        predicted = predictions(pinned, 10)
        shared = sorted(set(baseline_prediction) & set(predicted))
        entry = compare(baseline, values, TEXTURE_BANDS)
        entry["predictions_changed_vs_default"] = sum(
            baseline_prediction[k] != predicted[k] for k in shared)
        entry["predictions_compared"] = len(shared)
        report["pinned_grid"][pin] = entry
        print(f"  pinned at {pin:3} m: mean |delta| g_contrast "
              f"{entry['g_contrast']}, predictions changed "
              f"{entry['predictions_changed_vs_default']}/{len(shared)}")

    print("\nrequest scale of the delivered dashboard raster, by AOI width:")
    report["delivered_render_scales"] = delivered_render_scales()

    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {OUTPUT.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", default="Anuppur")
    parser.add_argument("--season", default="winter", choices=sorted(config.SEASONS))
    parser.add_argument("--points", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
