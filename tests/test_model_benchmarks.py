from backend.config import MODEL_METADATA, TRAINING_SCHEMA_VERSION


def test_random_forest_badge_uses_current_whole_mp_benchmark():
    rf = MODEL_METADATA["rf"]

    assert "internal_accuracy" not in rf
    assert rf["benchmark"]["training_schema_version"] == TRAINING_SCHEMA_VERSION
    assert (
        rf["benchmark"]["metric"]
        == "whole_mp_external_five_class_overall_accuracy"
    )
    assert rf["benchmark"]["winter"] == 78.04470440777105
    assert rf["benchmark"]["summer"] == 62.02723146747352
