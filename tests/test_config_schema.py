from backend.config import (
    BANDS,
    BEFORE_TRAINING_TABLE,
    EXPECTED_ASSET_LABELS,
    FEATURE_COLLECTIONS,
    LAND_COVER_CLASSES,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)


def test_six_class_mapping_is_fixed():
    assert {key: value["name"] for key, value in LAND_COVER_CLASSES.items()} == {
        0: "Forest",
        1: "Water",
        2: "Buildings",
        3: "Soil",
        4: "Sand",
        5: "Agriculture",
    }


def test_verified_19_band_cut_is_active():
    assert BANDS == [
        "B3",
        "B11",
        "B12",
        "BSI",
        "UI",
        "IBI",
        "SWIRratio",
        "BAEI",
        "g_contrast",
        "g_ent",
        "g_var",
        "g_idm",
        "g_diss",
        "g_asm",
        "VV",
        "VH",
        "VVVH",
        "s_contrast",
        "s_var",
    ]


def test_training_populations_are_complete():
    assert TRAINING_SCHEMA_VERSION == "six-class-19-band-v1"
    assert BEFORE_TRAINING_TABLE.endswith("/district_train_table")
    assert set(FEATURE_COLLECTIONS) == {
        "forest",
        "water",
        "buildings",
        "soil",
        "soil_mp",
        "sand_mp",
        "agriculture",
    }
    assert EXPECTED_ASSET_LABELS == {
        "forest": 0,
        "water": 1,
        "buildings": 2,
        "soil": 3,
        "soil_mp": 3,
        "sand_mp": 4,
        "agriculture": 5,
    }
    assert all("/water_points" not in path for path in FEATURE_COLLECTIONS.values())


def test_season_windows_are_exact():
    assert SEASONS == {
        "winter": {"start": "2025-01-01", "end": "2025-02-28"},
        "summer": {"start": "2025-03-31", "end": "2025-04-30"},
    }
