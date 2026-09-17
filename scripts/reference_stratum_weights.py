#!/usr/bin/env python3
"""Measure the area of every consensus-reference class in every MP district.

The statewide sample is stratified: 20 points per class per district, so every
class is over-represented wherever it is rare. An unweighted confusion matrix
built from that design does not estimate map-wide accuracy or class area -- it
estimates the accuracy of a population that does not exist. The Olofsson et al.
(2014) estimator repairs that, but it needs one number per stratum: how much
ground that stratum actually covers.

A stratum here is (district, reference class), matching how the sample was
drawn. This script measures each one by summing ee.Image.pixelArea() under the
class mask, rather than by histogramming the class image: a masked-area sum
aggregates correctly when Earth Engine works at a coarser scale, whereas a
categorical image aggregated by mean would blend class codes into values that
mean nothing.

Checkpointed per district-season so a rate-limit or dropped socket costs one
district, not the run.
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

from backend.config import SEASONS, TRAINING_SCHEMA_VERSION  # noqa: E402
from backend.gee_classifier import init_ee  # noqa: E402
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    madhya_pradesh_districts,
)

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "mp_reference_stratum_areas.json"


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
            time.sleep(5 * attempt)
    raise error


def district_stratum_areas(district, season, scale):
    """Area in m^2 of every reference class, plus the district's own area."""
    dates = SEASONS[season]
    region = district.geometry()
    reference = build_reference_image(
        region, season, dates["start"], dates["end"])
    # One grouped pass rather than five masked bands. The five-band version
    # evaluates the same reference image five times and times out on the larger
    # districts once the scale drops to 10 m; grouping by the class value
    # measures exactly the same masked areas in one pass. Masked pixels carry no
    # group and are dropped, which is the behaviour the five masks had.
    grouped = ee.Image.pixelArea().addBands(reference).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=2,
        bestEffort=False,
    )
    areas = {str(value): 0.0 for value in REFERENCE_LABELS}
    for group in ee.Dictionary(grouped).getInfo()["groups"]:
        if int(group["class"]) in REFERENCE_LABELS:
            areas[str(int(group["class"]))] = group["sum"]
    areas["district_area_m2"] = region.area(maxError=100).getInfo()
    return areas


def district_stratum_areas_tiled(district, season, scale, tiles=2):
    """The same measurement over a `tiles` x `tiles` grid, summed.

    The largest districts time out at 10 m even in one grouped pass. Splitting
    the bounding box and summing the pieces is arithmetically identical -- the
    strata partition the district and pixel areas add -- and each piece is a
    request Earth Engine will finish.
    """
    region = district.geometry()
    bounds = region.bounds().coordinates().get(0).getInfo()
    lons = [point[0] for point in bounds]
    lats = [point[1] for point in bounds]
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)
    totals = {str(value): 0.0 for value in REFERENCE_LABELS}
    for row in range(tiles):
        for column in range(tiles):
            cell = ee.Geometry.Rectangle([
                west + (east - west) * column / tiles,
                south + (north - south) * row / tiles,
                west + (east - west) * (column + 1) / tiles,
                south + (north - south) * (row + 1) / tiles,
            ], proj="EPSG:4326", geodesic=False)
            piece = ee.Feature(region.intersection(cell, maxError=10))
            if piece.geometry().area(maxError=100).getInfo() <= 0:
                continue
            measured = district_stratum_areas(piece, season, scale)
            measured.pop("district_area_m2")
            for key, value in measured.items():
                totals[key] += value
    totals["district_area_m2"] = region.area(maxError=100).getInfo()
    return totals


def merge(args):
    """Fold shard outputs into one file, asserting they agree on the scale."""
    merged = None
    for path in args.merge:
        piece = json.loads(path.read_text())
        if merged is None:
            merged = piece
            continue
        if piece["scale_m"] != merged["scale_m"]:
            raise SystemExit(f"{path} was measured at {piece['scale_m']} m")
        merged["strata"].update(piece["strata"])
    _write_json(args.output, merged)
    print(f"merged {len(args.merge)} shards into {args.output} "
          f"({len(merged['strata'])} district-seasons)")


def run(args):
    if args.merge:
        return merge(args)
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    _call_with_retry("initialize", init_ee, args.retries)
    districts = madhya_pradesh_districts()
    names = _call_with_retry(
        "district-list",
        lambda: districts.aggregate_array("ADM2_NAME").sort().getInfo(),
        args.retries,
    )
    output = args.output
    state = json.loads(output.read_text()) if output.exists() else {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "scale_m": args.scale,
        "reference_labels": REFERENCE_LABELS,
        "district_count_total": len(names),
        "strata": {},
    }
    if state.get("scale_m") != args.scale:
        raise SystemExit(
            f"{output} was measured at {state.get('scale_m')} m; rerun with "
            f"--scale {state.get('scale_m')} or start a new output.")
    if args.districts:
        names = [name for name in names if name in set(args.districts)]
        print(f"restricted to {len(names)} named districts", flush=True)
    if args.shards > 1:
        # Shard by index so each worker owns a disjoint district list and its
        # own checkpoint file; merge with --merge once they all land.
        names = [name for index, name in enumerate(names)
                 if index % args.shards == args.shard]
        print(f"shard {args.shard}/{args.shards}: {len(names)} districts",
              flush=True)
    for season in SEASONS:
        for name in names:
            key = f"{season}:{name}"
            if key in state["strata"]:
                continue
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            try:
                # Two attempts, not `--retries`: a district that times out at
                # this scale will time out again, and each attempt costs several
                # minutes before the server gives up. The tiled path below is
                # what actually finishes it, so get there quickly.
                areas = _call_with_retry(
                    key,
                    lambda: district_stratum_areas(district, season, args.scale),
                    2,
                )
            except RETRYABLE_ERRORS as whole_district:
                # A timeout here is a size problem, not a data problem: the same
                # measurement over pieces of the same district adds to the same
                # total. Say so in the log rather than losing the district.
                print(f"tiling {key} after {whole_district}", flush=True)
                areas = _call_with_retry(
                    f"{key} tiled",
                    lambda: district_stratum_areas_tiled(
                        district, season, args.scale, args.tiles),
                    args.retries,
                )
            state["strata"][key] = {
                "season": season,
                "district": name,
                "district_area_m2": areas.pop("district_area_m2"),
                "class_area_m2": {
                    REFERENCE_LABELS[int(value)]: areas.get(str(value)) or 0.0
                    for value in REFERENCE_LABELS
                },
            }
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(output, state)
            covered = sum(state["strata"][key]["class_area_m2"].values())
            print(f"landed {key} masked={covered / 1e6:9.1f} km2 "
                  f"({covered / state['strata'][key]['district_area_m2']:.1%} "
                  f"of district)", flush=True)
    print(f"wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--districts", nargs="*", default=None,
                        help="measure only these district names")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", type=Path, nargs="*", default=None,
                        help="merge these shard outputs into --output and exit")
    parser.add_argument("--tiles", type=int, default=3,
                        help="grid used when a whole district times out")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
