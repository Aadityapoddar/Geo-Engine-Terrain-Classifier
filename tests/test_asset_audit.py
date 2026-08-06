import pytest

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
    validate_audit_record(valid_record(), expected_label=5)


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
