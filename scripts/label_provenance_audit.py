#!/usr/bin/env python3
"""Describe the labelled training points: distribution, spacing, clustering.

The audit's complaint about label provenance is partly answerable from the data
and partly not. What the assets can be made to say:

  * how many points per class, and whether the stated 1,000 each is true
  * where they are -- bounding box, centroid, and the share falling inside
    Jabalpur district
  * how far apart they are, as the nearest-neighbour distance distribution.
    This is the number that decides whether a 100 m leakage buffer is doing
    anything: if the median spacing between training points is far below 100 m,
    the points are clustered tightly enough that a random split would routinely
    place neighbours on both sides, which is the mechanism behind the inflated
    94% random-split accuracy the paper reports.
  * how clustered, as the share of points within 100 m / 500 m / 1 km of
    another point of the same class
  * which district each point actually falls in. This is not bookkeeping: the
    points are described throughout as coming from Jabalpur district, and 872
    of the 5,000 do not. They sit in Katni, Dindori and Mandla, all of which
    were being scored as independent test districts at the same time.

What the assets cannot say, because the property does not exist on them, is who
interpreted each point, against which image and date, and whether a second
interpreter checked any of it. Those five collections carry exactly two
properties, `system:index` and `label`. That absence is itself the finding and
belongs in the paper's limitations rather than being filled in from memory.
"""

import argparse
import json
import sys
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    EXPECTED_ASSET_LABELS,
    FEATURE_COLLECTIONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import init_ee, merge_feature_collections  # noqa: E402
from evaluation.references import madhya_pradesh_districts  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "training_label_provenance.json"
NEIGHBOUR_BANDS_M = (100, 500, 1000)
MAX_NEIGHBOUR_SEARCH_M = 20000


def _nearest_neighbour_distances(collection):
    """Distance from each point to the closest other point in the collection."""
    join = ee.Join.saveAll(
        matchesKey="neighbours",
        measureKey="distance",
        ordering="distance",
        ascending=True,
        outer=True,
    )
    matched = join.apply(
        primary=collection,
        secondary=collection,
        condition=ee.Filter.withinDistance(
            distance=MAX_NEIGHBOUR_SEARCH_M,
            leftField=".geo",
            rightField=".geo",
            maxError=1,
        ),
    )

    def _closest(feature):
        # The first match is the point itself at distance 0, so take the second.
        neighbours = ee.List(feature.get("neighbours"))
        nearest = ee.Algorithms.If(
            neighbours.size().gt(1),
            ee.Feature(neighbours.get(1)).get("distance"),
            None,
        )
        return ee.Feature(None, {"nn_m": nearest})

    return matched.map(_closest).filter(ee.Filter.notNull(["nn_m"]))


def _containing_districts(collection, districts):
    """Histogram of the district each point falls in."""
    joined = ee.Join.saveFirst("district").apply(
        primary=collection,
        secondary=districts,
        condition=ee.Filter.intersects(
            leftField=".geo", rightField=".geo", maxError=10),
    )
    return joined.map(lambda feature: feature.set(
        "district_name",
        ee.Feature(feature.get("district")).get("ADM2_NAME"),
    )).aggregate_histogram("district_name")


def audit_class(name, path, jabalpur, districts):
    collection = ee.FeatureCollection(path)
    distances = _nearest_neighbour_distances(collection)
    percentiles = distances.reduceColumns(
        ee.Reducer.percentile([0, 5, 25, 50, 75, 95, 100]), ["nn_m"])
    within = {
        f"within_{band}m": distances.filter(
            ee.Filter.lt("nn_m", band)).size()
        for band in NEIGHBOUR_BANDS_M
    }
    payload = ee.Dictionary({
        "size": collection.size(),
        "labels": collection.aggregate_histogram("label"),
        "properties": collection.first().propertyNames(),
        "bounds": collection.geometry().bounds(1).coordinates(),
        "inside_jabalpur": collection.filterBounds(jabalpur).size(),
        "containing_districts": _containing_districts(collection, districts),
        "distance_outside_jabalpur_km": collection
        .filter(ee.Filter.bounds(jabalpur).Not())
        .map(lambda feature: feature.set(
            "d_km", feature.geometry().distance(jabalpur, 10).divide(1000)))
        .reduceColumns(ee.Reducer.percentile([50, 90, 100]), ["d_km"]),
        "nn_percentiles_m": percentiles,
        "nn_measured": distances.size(),
        **within,
    }).getInfo()
    payload["asset"] = path
    payload["class"] = name
    payload["expected_label"] = EXPECTED_ASSET_LABELS[name]
    return payload


def run(args):
    init_ee()
    districts = madhya_pradesh_districts()
    jabalpur = districts.filter(
        ee.Filter.eq("ADM2_NAME", "Jabalpur")).geometry()
    output = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "note": (
            "These collections carry only system:index and label. Interpreter "
            "identity, source image, interpretation date and any second-"
            "interpreter check are not recorded in the assets and cannot be "
            "reconstructed from them."
        ),
        "classes": {},
    }
    for name, path in FEATURE_COLLECTIONS.items():
        record = audit_class(name, path, jabalpur, districts)
        output["classes"][name] = record
        percentiles = record["nn_percentiles_m"]
        print(
            f"{name:12s} n={record['size']:5d} "
            f"label={sorted(record['labels'])} "
            f"in_jabalpur={record['inside_jabalpur']:5d} "
            f"nn median={percentiles.get('p50', float('nan')):8.1f} m "
            f"<100m={record['within_100m']:5d} "
            f"<500m={record['within_500m']:5d}",
            flush=True,
        )
    elsewhere = {}
    for record in output["classes"].values():
        for district, count in record["containing_districts"].items():
            if district != "Jabalpur":
                elsewhere[district] = elsewhere.get(district, 0) + count
    output["points_outside_jabalpur"] = dict(
        sorted(elsewhere.items(), key=lambda item: -item[1]))
    print(f"\npoints outside Jabalpur district: {output['points_outside_jabalpur']}")
    total = sum(record["size"] for record in output["classes"].values())
    output["total_points"] = total
    merged = merge_feature_collections(FEATURE_COLLECTIONS.values())
    output["merged_size"] = merged.size().getInfo()
    print(f"\ntotal {total} points across {len(output['classes'])} classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
