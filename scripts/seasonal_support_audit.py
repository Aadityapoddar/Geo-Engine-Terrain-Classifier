#!/usr/bin/env python3
"""Measure what the two seasons actually rest on, per pixel rather than per scene.

The manuscript argued that matched 59-day windows and similar per-district
granule counts make the seasonal comparison a comparison of phenology. Granule
counts are a scene-level statistic: they say how many images intersected a
district, not how many times a given pixel was seen through a gap in the cloud.
Two further asymmetries were asserted away rather than measured. The percentile
bounds that map every unbounded band onto [0, 1] were fitted on the winter
training sample and then applied unchanged to summer, so a summer distribution
that runs hotter is clipped against a winter tail. And Sentinel-1 has its own
per-pixel revisit, which no optical count describes.

This measures all three, per district and per season:

  * s2_valid_observations -- per-pixel count of Sentinel-2 observations
    surviving the scene-level cloud filter and the QA60 mask, summarised over
    the district (mean, and the 5th/50th percentiles);
  * s1_observations -- the same for the descending-orbit radar stack;
  * saturated_fraction -- for each rescaled band, the share of district pixels
    clipped to exactly 0 or exactly 1 by the winter bounds in
    backend.gee_classifier.FEATURE_RANGES.

Nothing here changes a reported accuracy. It exists so the seasonal difference
can be attributed with the evidence in hand rather than by assumption.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ee  # noqa: E402

from backend.config import BANDS, SEASONS  # noqa: E402
from backend.gee_classifier import (  # noqa: E402
    FEATURE_RANGES,
    build_sentinel_composite,
    init_ee,
    mask_s2_clouds,
    sentinel1_collection,
)
from evaluation.references import madhya_pradesh_districts  # noqa: E402
from evaluation.splits import district_splits  # noqa: E402

OUT = REPO / "doc" / "assets" / "seasonal_support_audit.json"
CLOUD_THRESHOLD = 15
# Coarser than the 10 m classification grid on purpose. These are distributional
# summaries over a whole district, and at 10 m a single district-season is a
# multi-minute request for a number that does not move in the third decimal.
SCALE_M = 300
# Only the rescaled bands can saturate; the rest are already bounded ratios.
SATURATING = [name for name in FEATURE_RANGES if name in BANDS]


def observation_counts(region, start, end):
    """Per-pixel counts of usable optical and radar observations."""
    optical = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
        .select("B8")
        .count()
        .unmask(0)
        .rename("s2_valid_observations")
    )
    radar = (
        sentinel1_collection(region, start, end)
        .select("VV")
        .count()
        .unmask(0)
        .rename("s1_observations")
    )
    return optical.addBands(radar)


def saturation(region, start, end):
    """Share of pixels each rescaled band clips to 0 or 1 under the winter bounds."""
    composite = build_sentinel_composite(
        geometry=region, start_date=start, end_date=end,
        cloud_threshold=CLOUD_THRESHOLD, allow_temporal_fallback=False)
    flags = []
    for name in SATURATING:
        band = composite.select(name)
        clipped = band.lte(0).Or(band.gte(1)).rename(f"sat_{name}")
        flags.append(clipped)
    return ee.Image.cat(flags)


def audit(district_names, seasons):
    districts = madhya_pradesh_districts()
    results = {}
    for season in seasons:
        dates = SEASONS[season]
        for name in district_names:
            key = f"{season}:{name}"
            region = districts.filter(ee.Filter.eq("ADM2_NAME", name)).geometry()
            counts = observation_counts(region, dates["start"], dates["end"])
            reducer = (ee.Reducer.mean()
                       .combine(ee.Reducer.percentile([5, 50]), sharedInputs=True))
            summary = counts.reduceRegion(
                reducer=reducer, geometry=region, scale=SCALE_M,
                maxPixels=1e10, bestEffort=True).getInfo()
            saturated = saturation(region, dates["start"], dates["end"]).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=SCALE_M,
                maxPixels=1e10, bestEffort=True).getInfo()
            results[key] = {
                "season": season,
                "district": name,
                "observations": summary,
                "saturated_fraction": saturated,
            }
            print(f"{key:24s} S2 mean "
                  f"{summary.get('s2_valid_observations_mean', float('nan')):5.1f} "
                  f"p5 {summary.get('s2_valid_observations_p5', float('nan')):5.1f}  "
                  f"S1 mean {summary.get('s1_observations_mean', float('nan')):5.1f}  "
                  f"max saturated "
                  f"{max(v for v in saturated.values() if v is not None):.4f}",
                  flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--districts", type=int, default=4,
                        help="how many test districts to audit, spread by name order")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    init_ee()
    test = district_splits()["test"]
    step = max(1, len(test) // args.districts)
    chosen = test[::step][:args.districts]
    print(f"auditing {len(chosen)} of {len(test)} test districts: "
          f"{', '.join(chosen)}\n")

    payload = {
        "scale_m": SCALE_M,
        "cloud_threshold_percent": CLOUD_THRESHOLD,
        "saturating_bands": SATURATING,
        "feature_ranges": {name: FEATURE_RANGES[name] for name in SATURATING},
        "bounds_fitted_on": "winter p99 of the labelled training sample",
        "districts": chosen,
        "results": audit(chosen, ("winter", "summer")),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
