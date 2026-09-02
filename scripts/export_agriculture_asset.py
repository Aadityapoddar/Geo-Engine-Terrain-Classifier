#!/usr/bin/env python3
"""Export the 1,000 Agriculture points to Earth Engine as a labelled asset.

`jabalpur_agriculture_points_updated` was imported from a drawn geometry and
carries no attributes at all: every feature has empty properties. That merges
into the training population without complaint, and then
`sampleRegions(properties=["label"])` plus the `notNull` filter drops all 1,000
points, so Agriculture trains on nothing and the class simply never appears in
a prediction. This writes a `_labelled` sibling rather than overwriting the
original import.

The label comes from `EXPECTED_ASSET_LABELS` so it cannot drift from the
backend inventory. Whatever a source feature carries -- a stale label from an
earlier inventory, a null, a missing key -- is overwritten: this collection is
Agriculture and nothing else.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import EE_ASSET_ROOT, EXPECTED_ASSET_LABELS  # noqa: E402
from backend.gee_classifier import init_ee  # noqa: E402

AGRICULTURE_LABEL = EXPECTED_ASSET_LABELS["agriculture"]
DEFAULT_SOURCE = REPO / "doc" / "assets" / "jabalpur_agriculture_points_1000.geojson"
DEFAULT_ASSET = f"{EE_ASSET_ROOT}/jabalpur_agriculture_points_labelled"
DEFAULT_SOURCE_ASSET = f"{EE_ASSET_ROOT}/jabalpur_agriculture_points_updated"


def labelled_features(payload):
    """Return the source features with a valid Agriculture label on every one.

    Returns (features, relabelled) so the caller can report how many features
    arrived carrying some other label instead of assuming the overwrite did
    nothing.
    """
    features = payload.get("features", [])
    if not features:
        raise ValueError("Source GeoJSON contains no features")

    geometry_types = {feature.get("geometry", {}).get("type") for feature in features}
    if geometry_types != {"Point"}:
        raise ValueError(f"Expected only Point geometries, got {sorted(geometry_types)}")

    relabelled = 0
    for feature in features:
        properties = feature.setdefault("properties", {}) or {}
        label = properties.get("label")
        if label is not None and label != AGRICULTURE_LABEL:
            relabelled += 1
        properties["label"] = AGRICULTURE_LABEL
        feature["properties"] = properties

    return features, relabelled


def _from_geojson(source, expected_count):
    """Build the collection by uploading 1,000 points as an inline expression."""
    payload = json.loads(Path(source).read_text())
    features, relabelled = labelled_features(payload)
    if expected_count is not None and len(features) != expected_count:
        raise ValueError(f"Expected {expected_count} features, got {len(features)}")
    print(f"source=geojson:{source}")
    print(f"features={len(features)} label={AGRICULTURE_LABEL} "
          f"relabelled={relabelled}")
    return ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point(feature["geometry"]["coordinates"]),
            {"label": AGRICULTURE_LABEL},
        )
        for feature in features
    ])


def _from_asset(source_asset, expected_count):
    """Relabel the geometries already in Earth Engine.

    The unlabelled import holds exactly the geometries we want; only the `label`
    property is missing. Setting it server-side keeps the export graph to a
    handful of nodes, where re-uploading 1,000 inline points builds an
    expression large enough that the batch scheduler sits on it for hours.
    """
    collection = ee.FeatureCollection(source_asset)
    size = collection.size().getInfo()
    if expected_count is not None and size != expected_count:
        raise ValueError(f"Expected {expected_count} features, got {size}")
    print(f"source=asset:{source_asset}")
    print(f"features={size} label={AGRICULTURE_LABEL} (set server-side)")
    # Unconditional: this collection is Agriculture and nothing else, so a
    # missing label and any stale label both resolve to the canonical value.
    return collection.map(
        lambda feature: feature.set("label", AGRICULTURE_LABEL)
    )


def export(source, asset, expected_count, wait, source_asset=None):
    init_ee()
    collection = (
        _from_asset(source_asset, expected_count) if source_asset
        else _from_geojson(source, expected_count)
    )

    task = ee.batch.Export.table.toAsset(
        collection=collection,
        description="agriculture_points_labelled",
        assetId=asset,
    )
    task.start()
    print(f"started export task {task.id} -> {asset}")
    if not wait:
        return task.id

    while True:
        status = task.status()
        state = status["state"]
        print(f"  {state}", flush=True)
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            if state != "COMPLETED":
                raise RuntimeError(f"Export {state}: {status.get('error_message')}")
            break
        time.sleep(15)

    exported = ee.FeatureCollection(asset)
    histogram = exported.aggregate_histogram("label").getInfo()
    size = exported.size().getInfo()
    print(f"verified {asset}: size={size} labels={histogram}")
    if histogram != {str(AGRICULTURE_LABEL): size}:
        raise RuntimeError(f"Exported asset carries unexpected labels: {histogram}")
    return task.id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-asset", nargs="?", const=DEFAULT_SOURCE_ASSET, default=None,
        help="Relabel this existing asset instead of uploading the GeoJSON.")
    parser.add_argument("--asset", default=DEFAULT_ASSET)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()
    export(args.source, args.asset, args.expected_count, args.wait,
           args.source_asset)


def _self_check():
    """The relabel is the whole point of this script, so check it directly."""
    label = AGRICULTURE_LABEL
    features, relabelled = labelled_features({
        "features": [
            {"geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {"label": label}},
            {"geometry": {"type": "Point", "coordinates": [3, 4]}, "properties": {"label": 5}},
            {"geometry": {"type": "Point", "coordinates": [5, 6]}, "properties": {}},
            {"geometry": {"type": "Point", "coordinates": [7, 8]}},
        ]
    })
    assert [f["properties"]["label"] for f in features] == [label] * 4
    assert relabelled == 1
    for bad in ({"features": []},
                {"features": [{"geometry": {"type": "Polygon", "coordinates": []}}]}):
        try:
            labelled_features(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected rejection for {bad}")
    print("self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
