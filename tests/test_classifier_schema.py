from unittest.mock import patch

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
