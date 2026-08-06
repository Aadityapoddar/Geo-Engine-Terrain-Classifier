from evaluation.references import (
    DW_MIN_PROBABILITY,
    GHSL_MIN_BUILT_SQM,
    PROJECT_TO_REFERENCE_LABEL,
    REFERENCE_LABELS,
    WORLD_CEREAL_SEASON,
    collapse_project_prediction,
)


def test_reference_schema_and_thresholds_are_fixed():
    assert DW_MIN_PROBABILITY == 0.70
    assert GHSL_MIN_BUILT_SQM == 50
    assert REFERENCE_LABELS == {
        0: "Forest",
        1: "Water",
        2: "Buildings",
        3: "Bare",
        4: "Agriculture",
    }
    assert PROJECT_TO_REFERENCE_LABEL == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
    assert WORLD_CEREAL_SEASON == {
        "winter": "tc-wintercereals",
        "summer": "tc-maize-main",
    }


def test_soil_and_sand_collapse_only_for_external_reference():
    assert collapse_project_prediction([0, 1, 2, 3, 4, 5]) == [0, 1, 2, 3, 3, 4]
