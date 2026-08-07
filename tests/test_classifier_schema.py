import json
from unittest.mock import patch

import ee
import pytest

from backend import gee_classifier as gc


def test_cache_key_contains_training_schema_version():
    key = gc._classifier_cache_key("RF", "2025-01-01", "2025-02-28", 15)
    assert key == ("rf", "2025-01-01", "2025-02-28", 15, "six-class-19-band-v1")


@patch.object(gc.ee.Classifier, "smileKNN")
def test_knn_factory_keeps_k_five(factory):
    gc.make_classifier("knn")
    factory.assert_called_once_with(k=5)


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
