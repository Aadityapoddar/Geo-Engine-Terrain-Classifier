from evaluation.references import (
    DW_MIN_PROBABILITY,
    GHSL_MIN_BUILT_SQM,
    PROJECT_TO_REFERENCE_LABEL,
    REFERENCE_LABELS,
    WORLD_CEREAL_ID,
    WORLD_CEREAL_PRODUCT,
    WORLD_CEREAL_SEASON,
    collapse_project_prediction,
)


def test_reference_schema_and_thresholds_are_fixed():
    assert DW_MIN_PROBABILITY == 0.70
    assert GHSL_MIN_BUILT_SQM == 50
    assert REFERENCE_LABELS == {
        0: "Vegetation",
        1: "Water",
        2: "Built Area",
        3: "Open Land",
        4: "Agriculture",
    }
    assert PROJECT_TO_REFERENCE_LABEL == {
        "before": {0: 0, 1: 1, 2: 2, 3: 3, 4: 3},
        "after": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    }
    assert WORLD_CEREAL_SEASON == {
        "winter": "tc-wintercereals",
        "summer": "tc-maize-main",
    }
    assert WORLD_CEREAL_ID == "ESA/WorldCereal/2021/MODELS/v100"
    assert WORLD_CEREAL_PRODUCT == "temporarycrops"


def test_collapse_is_per_condition():
    # Before: Soil (3) and Sand (4) both read as Bare; Agriculture was never learned.
    assert collapse_project_prediction([0, 1, 2, 3, 4], "before") == [0, 1, 2, 3, 3]
    # After: Barren Land (3) reads as Bare; Agriculture (4) maps to itself.
    assert collapse_project_prediction([0, 1, 2, 3, 4], "after") == [0, 1, 2, 3, 4]
