import pytest

from evaluation import assets
from evaluation.assets import validate_audit_record


def valid_record():
    return {
        "name": "agriculture",
        "size": 1000,
        "geometry_types": {"Point": 1000},
        "label_histogram": {"5": 1000},
        "seasonal_valid_samples": {"winter": 997, "summer": 995},
    }


def test_valid_asset_record_passes():
    validate_audit_record(valid_record(), expected_label=5, expected_count=1000)


def test_asset_with_wrong_expected_count_fails():
    record = valid_record()
    record["size"] = 407
    with pytest.raises(ValueError, match="expected 1000 features, got 407"):
        validate_audit_record(record, expected_label=5, expected_count=1000)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("size", 0, "is empty"),
        ("geometry_types", {"Polygon": 1}, "non-Point"),
        ("label_histogram", {"4": 1000}, "expected label 5"),
        ("seasonal_valid_samples", {"winter": 0, "summer": 995}, "zero valid winter"),
    ],
)
def test_invalid_asset_record_fails(field, value, message):
    record = valid_record()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        validate_audit_record(record, expected_label=5)


def test_full_audit_initializes_earth_engine_before_loading_assets(monkeypatch):
    events = []

    monkeypatch.setattr(assets, "init_ee", lambda: events.append("initialized"))

    def feature_collection(path):
        assert events == ["initialized"]
        return path

    monkeypatch.setattr(assets.ee, "FeatureCollection", feature_collection)
    monkeypatch.setattr(
        assets,
        "audit_feature_collection",
        lambda name, collection, expected_label=None, expected_count=None: {
            "name": name
        },
    )

    records = assets.audit_all_training_assets()

    assert len(records) == 1 + len(assets.FEATURE_COLLECTIONS)
