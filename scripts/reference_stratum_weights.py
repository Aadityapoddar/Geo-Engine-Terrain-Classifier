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
    pixel_area = ee.Image.pixelArea()
    stack = ee.Image.cat([
        pixel_area.updateMask(reference.eq(value)).rename(str(value))
        for value in REFERENCE_LABELS
    ])
    measured = stack.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=8,
        bestEffort=False,
    )
    return ee.Dictionary(measured).set(
        "district_area_m2", region.area(maxError=100)).getInfo()


def run(args):
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
    for season in SEASONS:
        for name in names:
            key = f"{season}:{name}"
            if key in state["strata"]:
                continue
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            areas = _call_with_retry(
                key,
                lambda: district_stratum_areas(district, season, args.scale),
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
