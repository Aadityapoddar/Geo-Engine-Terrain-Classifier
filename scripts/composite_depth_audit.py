#!/usr/bin/env python3
"""Count the observations actually reaching each district-season composite.

Two of the audit's method objections are really the same question: is the
winter-to-summer accuracy gap a seasonal effect, or a data-availability effect?
Equal-length windows (backend/config.SEASONS) remove the most obvious way for
the two to be confounded, but equal length is not equal depth -- a 59-day
pre-monsoon window and a 59-day winter window can still differ in how many
scenes clear the 15% cloud filter, and a thinner median is a noisier median.

So measure it rather than assert it. For every district and season this records:

  s2_scenes          Sentinel-2 granules passing the cloud filter in-window
  s2_valid_fraction  share of district pixels with at least one unmasked
                     observation, i.e. how much of the district the optical
                     composite actually covers
  s1_scenes          Sentinel-1 IW descending dual-pol scenes in-window
  s1_missing         True where the radar window is empty; with the experiment
                     path's temporal fallback disabled these are the districts
                     whose points drop out, and the number belongs in the paper
                     rather than in a silent substitution

Cheap compared with the classification runs -- metadata counts and one masked
fraction per district-season.
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
from backend.gee_classifier import (  # noqa: E402
    init_ee,
    mask_s2_clouds,
    sentinel1_collection,
)
from evaluation.references import madhya_pradesh_districts  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "mp_composite_depth.json"
CLOUD_THRESHOLD = 15


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


def district_depth(district, season, scale):
    dates = SEASONS[season]
    region = district.geometry()
    optical = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(dates["start"], dates["end"])
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
    )
    masked = optical.map(mask_s2_clouds)
    # unmask(0) then mean gives the share of the district that survived cloud
    # masking in at least one scene; the median composite is undefined exactly
    # where this is zero.
    valid = masked.select("B8").count().gt(0).unmask(0).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=8,
    ).get("B8")
    radar = sentinel1_collection(region, dates["start"], dates["end"])
    return ee.Dictionary({
        "s2_scenes": optical.size(),
        "s2_valid_fraction": ee.Number(valid),
        "s1_scenes": radar.size(),
    }).getInfo()


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
        "seasons": SEASONS,
        "cloud_threshold": CLOUD_THRESHOLD,
        "scale_m": args.scale,
        "records": {},
    }
    for season in SEASONS:
        for name in names:
            key = f"{season}:{name}"
            if key in state["records"]:
                continue
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            depth = _call_with_retry(
                key, lambda: district_depth(district, season, args.scale),
                args.retries)
            depth.update({
                "season": season,
                "district": name,
                "s1_missing": depth["s1_scenes"] == 0,
            })
            state["records"][key] = depth
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(output, state)
            print(f"landed {key} s2={depth['s2_scenes']:4d} "
                  f"valid={depth['s2_valid_fraction']:.3f} "
                  f"s1={depth['s1_scenes']:3d}", flush=True)
    summarise(state)
    print(f"wrote {output}", flush=True)


def summarise(state):
    for season in state["seasons"]:
        rows = [record for record in state["records"].values()
                if record["season"] == season]
        if not rows:
            continue
        count = len(rows)
        print(f"\n{season}: {count} districts")
        for field in ("s2_scenes", "s2_valid_fraction", "s1_scenes"):
            values = sorted(record[field] for record in rows)
            mean = sum(values) / count
            print(f"  {field:18s} mean {mean:8.3f}  min {values[0]:8.3f}  "
                  f"median {values[count // 2]:8.3f}  max {values[-1]:8.3f}")
        print(f"  districts with no in-window Sentinel-1: "
              f"{sum(record['s1_missing'] for record in rows)}")
        print(f"  districts with incomplete optical cover (<99%): "
              f"{sum(record['s2_valid_fraction'] < 0.99 for record in rows)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()
    if args.summarise_only:
        summarise(json.loads(args.output.read_text()))
        return
    run(args)


if __name__ == "__main__":
    main()
