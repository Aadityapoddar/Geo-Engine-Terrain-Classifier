import json
from unittest.mock import patch
from backend.config import TRAINING_SCHEMA_VERSION

import ee
import pytest

from backend import gee_classifier as gc


def test_cache_key_contains_training_schema_version():
    key = gc._classifier_cache_key("RF", "2025-01-01", "2025-03-01", 15, False)
    assert key == (
        "rf", "2025-01-01", "2025-03-01", 15, False, TRAINING_SCHEMA_VERSION)
    # A fallback-widened composite is different imagery, so it must not share
    # a cache entry with the strict one.
    assert key != gc._classifier_cache_key(
        "RF", "2025-01-01", "2025-03-01", 15, True)


@patch.object(gc.ee.Classifier, "smileKNN")
def test_factory_uses_the_configured_settings(factory):
    """make_classifier must build from config, not from a second copy.

    The settings used to be written out twice, here and in MODEL_METADATA, and
    a hyperparameter search that touched one left the other stale. This pins
    the call to whatever config currently ships -- k was 5 and is now 9 -- so
    the test tracks a retune instead of failing on it.
    """
    gc.make_classifier("knn")
    factory.assert_called_once_with(**gc.classifier_parameters("knn"))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        gc.make_classifier("bogus")


def test_empty_collection_list_is_rejected():
    with pytest.raises(ValueError, match="At least one training collection"):
        gc.merge_feature_collections([])


def test_exportable_training_samples_retain_point_geometry():
    gc.init_ee()
    image = ee.Image.constant(list(range(len(gc.BANDS)))).rename(gc.BANDS)
    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([80.0, 23.0]), {"label": 0})
    ])

    samples = gc._sample_regions_with_geometry(image, points)
    encoded = json.dumps(ee.serializer.encode(samples), separators=(",", ":"))

    assert '"geometries":{"constantValue":true}' in encoded


def test_sentinel_collection_fallback_is_server_side(monkeypatch):
    class FakeNumber:
        def gt(self, value):
            return ("gt", value)

    class FakeCollection:
        def filterDate(self, *args):
            return self

        def filterBounds(self, *args):
            return self

        def filter(self, *args):
            return self

        def map(self, *args):
            return self

        def size(self):
            return FakeNumber()

    collection = FakeCollection()
    monkeypatch.setattr(gc.ee, "ImageCollection", lambda value: collection)
    monkeypatch.setattr(gc.ee.Filter, "lt", lambda *args: args)
    monkeypatch.setattr(
        gc.ee.Algorithms, "If", lambda condition, yes, no: yes, raising=False
    )

    assert gc._build_collection("geometry", "start", "end", 15) is collection
