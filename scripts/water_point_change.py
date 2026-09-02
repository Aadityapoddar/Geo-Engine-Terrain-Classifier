#!/usr/bin/env python3
"""Measure what changed between the Before and After Water training points.

The Before population preserves the original 1,000 Jabalpur water geometries in
`district_train_table`; the After population uses the repositioned
`jabalpur_water_points`. Both are 1,000 points, so a size check says nothing.
What matters is whether the points moved, and whether they moved onto water:
this compares nearest-neighbour displacement and scores each set against
per-season NDWI and the JRC global surface-water occurrence layer.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    BEFORE_TRAINING_TABLE,
    FEATURE_COLLECTIONS,
    SEASONS,
)
from backend.gee_classifier import _build_collection, init_ee  # noqa: E402

WATER_LABEL = 1
# JRC occurrence is a percentage of observations that saw water between 1984 and
# 2021. 25% keeps seasonal tanks and river channels, which a 90% threshold would
# throw away along with the noise.
OCCURRENCE_THRESHOLD = 25


def _water_sets():
    """Return Before and After water points stripped to bare geometries.

    `district_train_table` is a sampled table: it already carries NDWI, B3, VV
    and the rest as feature properties. sampleRegions keeps input properties, so
    a stored NDWI column silently shadows the NDWI band being sampled and the
    Before set reports the same value in every season. Dropping properties first
    is what makes the two sides comparable at all.
    """
    bare = lambda feature: ee.Feature(feature.geometry())  # noqa: E731
    before = (
        ee.FeatureCollection(BEFORE_TRAINING_TABLE)
        .filter(ee.Filter.eq("label", WATER_LABEL))
        .map(bare)
    )
    after = ee.FeatureCollection(FEATURE_COLLECTIONS["water"]).map(bare)
    return before, after


def _coordinates(collection):
    return collection.geometry().coordinates().getInfo()


def _displacement(before_coords, after_coords):
    """Nearest-neighbour distance from each After point to the Before set.

    Done in plain Python: 1,000 x 1,000 is a trivial scan, and it avoids
    depending on how ee.Join reports its measure key.
    ponytail: O(n^2) scan, fine at 1e6 pairs; use a KD-tree if the sets grow.
    """
    metres_per_degree = 111320.0
    distances = []
    for lon, lat in after_coords:
        scale = math.cos(math.radians(lat))
        best = min(
            (lon - blon) ** 2 * scale ** 2 + (lat - blat) ** 2
            for blon, blat in before_coords
        )
        distances.append(math.sqrt(best) * metres_per_degree)
    distances.sort()
    count = len(distances)
    # sampleRegions(geometries=True) returns the centre of the 10 m pixel it read,
    # not the point it was given, so a table round-trip displaces every point by
    # up to half a pixel diagonal. If the whole distribution sits under that
    # bound, the two sets are the same points and nothing was repositioned.
    pixel_snap_bound = math.hypot(10.0, 10.0) / 2
    maximum = distances[-1]
    return {
        "compared": count,
        "identical_points": sum(distance < 1.0 for distance in distances),
        "moved_over_100m": sum(distance > 100.0 for distance in distances),
        "mean_metres": round(sum(distances) / count, 2),
        "median_metres": round(distances[count // 2], 2),
        "max_metres": round(maximum, 2),
        "pixel_snap_bound_metres": round(pixel_snap_bound, 2),
        "explained_by_pixel_snapping": maximum <= pixel_snap_bound,
    }


def _on_water_fraction(points, season):
    """Fraction of points that land on water by NDWI and by JRC occurrence."""
    dates = SEASONS[season]
    region = points.geometry().bounds()
    composite = _build_collection(
        region, dates["start"], dates["end"], 15
    ).median()
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

    sampled = ndwi.addBands(occurrence).sampleRegions(
        collection=points, scale=10, tileScale=4
    ).filter(ee.Filter.notNull(["NDWI"]))

    return {
        "sampled": sampled.size(),
        # Positive NDWI is the standard open-water cut.
        "ndwi_positive": sampled.filter(ee.Filter.gt("NDWI", 0)).size(),
        "mean_ndwi": sampled.aggregate_mean("NDWI"),
        "jrc_water": sampled.filter(
            ee.Filter.gte("occurrence", OCCURRENCE_THRESHOLD)
        ).size(),
    }


def analyse():
    init_ee()
    before, after = _water_sets()
    payload = {
        "before_asset": BEFORE_TRAINING_TABLE,
        "after_asset": FEATURE_COLLECTIONS["water"],
        "counts": {"before": before.size(), "after": after.size()},
        "seasons": {},
    }
    for season in SEASONS:
        payload["seasons"][season] = {
            "before": _on_water_fraction(before, season),
            "after": _on_water_fraction(after, season),
        }
    resolved = ee.Dictionary(payload).getInfo()
    resolved["displacement"] = _displacement(
        _coordinates(before), _coordinates(after)
    )
    return resolved


def _percent(part, whole):
    return None if not whole else round(100.0 * part / whole, 2)


def summarise(payload):
    for season, sides in payload["seasons"].items():
        for condition, values in sides.items():
            values["ndwi_positive_percent"] = _percent(
                values["ndwi_positive"], values["sampled"])
            values["jrc_water_percent"] = _percent(
                values["jrc_water"], values["sampled"])
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "doc" / "assets" / "water_point_change.json")
    args = parser.parse_args()
    payload = summarise(analyse())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
