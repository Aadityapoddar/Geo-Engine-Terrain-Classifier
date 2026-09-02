import json

import pytest

from backend.config import TRAINING_SCHEMA_VERSION
from scripts.aggregate_mp_external_shards import aggregate


def _matrix(correct):
    value = [[0] * 5 for _ in range(5)]
    value[0][0] = correct
    return value


def _worker(path, index, district, schema=TRAINING_SCHEMA_VERSION):
    matrices = {
        f"{condition}-{model}": _matrix(index + 1)
        for condition in ("before", "after")
        for model in ("rf", "svm", "gtb", "cart", "knn")
    }
    shards = {
        f"{season}:{district}": {
            "season": season,
            "district": district,
            "sample_count": index + 1,
            "matrices": matrices,
        }
        for season in ("winter", "summer")
    }
    path.write_text(json.dumps({
        "training_schema_version": schema,
        "district_count_total": 2,
        "assigned_districts": [district],
        "shards": shards,
    }))


def test_aggregate_requires_and_combines_every_district(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _worker(first, 0, "A")
    _worker(second, 1, "B")

    result = aggregate([first, second])

    assert result["district_count"] == 2
    assert result["shard_count"] == 4
    assert len(result["runs"]) == 20
    assert result["runs"][0]["metrics"]["external_five_class"][
        "sample_count"
    ] == 3


def test_aggregate_rejects_incomplete_worker_set(tmp_path):
    first = tmp_path / "first.json"
    _worker(first, 0, "A")

    with pytest.raises(ValueError, match="cover 1/2"):
        aggregate([first])


def test_aggregate_refuses_shards_from_a_different_class_schema(tmp_path):
    # Class 3 was Soil under v1 and is Barren Land now. Summing the two would
    # produce a confusion matrix whose rows mean different things.
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _worker(first, 0, "A")
    _worker(second, 1, "B", schema="six-class-19-band-v1")

    with pytest.raises(ValueError, match="cannot be summed"):
        aggregate([first, second])


def test_aggregate_refuses_unstamped_shards(tmp_path):
    first = tmp_path / "first.json"
    _worker(first, 0, "A")
    payload = json.loads(first.read_text())
    del payload["training_schema_version"]
    first.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unstamped"):
        aggregate([first])
