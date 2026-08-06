"""Training-population selection and live Earth Engine asset audits."""

import ee

from backend.config import (
    BEFORE_TRAINING_TABLE,
    EXPECTED_ASSET_LABELS,
    FEATURE_COLLECTIONS,
    SEASONS,
)
from backend.gee_classifier import (
    init_ee,
    merge_feature_collections,
    sample_training_points,
)


def training_points(condition):
    """Return the preserved Before population or all current After assets."""
    if condition == "before":
        return ee.FeatureCollection(BEFORE_TRAINING_TABLE)
    if condition == "after":
        return merge_feature_collections(FEATURE_COLLECTIONS.values())
    raise ValueError(f"Unknown training condition: {condition}")


def validate_audit_record(record, expected_label=None):
    """Reject records that cannot safely participate in model training."""
    name = record.get("name", "asset")
    if record.get("size", 0) <= 0:
        raise ValueError(f"{name} is empty")

    geometry_types = record.get("geometry_types", {})
    if set(geometry_types) != {"Point"}:
        raise ValueError(f"{name} contains non-Point geometries: {geometry_types}")

    histogram = record.get("label_histogram", {})
    if not histogram:
        raise ValueError(f"{name} has no labels")
    if expected_label is not None and set(histogram) != {str(expected_label)}:
        raise ValueError(
            f"{name} expected label {expected_label}, got {sorted(histogram)}")

    for season, count in record.get("seasonal_valid_samples", {}).items():
        if count <= 0:
            raise ValueError(f"{name} has zero valid {season} samples")
    if set(record.get("seasonal_valid_samples", {})) != set(SEASONS):
        raise ValueError(f"{name} is missing seasonal sample counts")
    return record


def audit_feature_collection(name, collection, expected_label=None):
    """Collect counts, geometry/label integrity, bounds, and seasonal samples."""
    init_ee()
    collection = ee.FeatureCollection(collection)
    typed = collection.map(
        lambda feature: feature.set("_geometry_type", feature.geometry().type()))
    record = {
        "name": name,
        "size": collection.size().getInfo(),
        "geometry_types": typed.aggregate_histogram("_geometry_type").getInfo(),
        "label_histogram": collection.aggregate_histogram("label").getInfo(),
        "bounds": collection.geometry().bounds().coordinates().getInfo(),
        "seasonal_valid_samples": {},
    }
    for season, dates in SEASONS.items():
        samples = sample_training_points(
            collection, start_date=dates["start"], end_date=dates["end"])
        record["seasonal_valid_samples"][season] = samples.size().getInfo()
    return validate_audit_record(record, expected_label)


def audit_all_training_assets():
    """Audit the preserved Before table and every configured After source."""
    records = [
        audit_feature_collection(
            "before:district_train_table",
            ee.FeatureCollection(BEFORE_TRAINING_TABLE),
        )
    ]
    for name, path in FEATURE_COLLECTIONS.items():
        records.append(
            audit_feature_collection(
                f"after:{name}",
                ee.FeatureCollection(path),
                EXPECTED_ASSET_LABELS[name],
            )
        )
    return records
