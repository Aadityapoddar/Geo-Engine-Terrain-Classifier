import json
import re
from datetime import date
from pathlib import Path

import pytest

from backend.config import (
    BANDS,
    FULL_BANDS,
    BEFORE_TRAINING_TABLE,
    EXPECTED_ASSET_LABELS,
    FEATURE_COLLECTIONS,
    LAND_COVER_CLASSES,
    SEASON_WINDOW_DAYS,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)

COMPARISON = (Path(__file__).resolve().parents[1] / "doc" / "assets" /
              "band_stack_comparison_v4.json")


def test_five_class_mapping_is_fixed():
    assert {key: value["name"] for key, value in LAND_COVER_CLASSES.items()} == {
        0: "Vegetation",
        1: "Water",
        2: "Built Area",
        3: "Open Land",
        4: "Agriculture",
    }


def test_shipped_stack_is_the_one_the_districts_chose():
    """The band list is a measured result now, so pin it to the measurement.

    It used to be a literal in this file, which only proved that config still
    said what the test said. The stack is selected on the development districts
    and confirmed on the held-out ones; asserting against that artefact is what
    actually catches a config edited without a rerun, or a rerun nobody copied
    across.
    """
    assert BANDS, "no bands configured"
    assert set(BANDS) <= set(FULL_BANDS), (
        f"{sorted(set(BANDS) - set(FULL_BANDS))} are not in the 27-band "
        "superset the composite computes")
    assert len(set(BANDS)) == len(BANDS), "the band list repeats an entry"

    # The selection artefact and the rule that reads it live in the research
    # tree, which a production checkout does not carry. The assertions above
    # still guard the shipped config; this one is a bonus when the evidence is
    # present.
    try:
        from scripts.adopt_band_stack import select_stack
    except ImportError:
        pytest.skip("scripts/ is not part of this checkout")
    if not COMPARISON.exists():
        pytest.skip(f"{COMPARISON.name} has not been generated")
    comparison = json.loads(COMPARISON.read_text())
    chosen, _ = select_stack(comparison)
    assert BANDS == comparison["stacks"][chosen], (
        f"config ships {len(BANDS)} bands but the development districts "
        f"selected {chosen} ({len(comparison['stacks'][chosen])} bands)")


def test_schema_version_records_the_band_count():
    """A stack change that leaves the schema alone would silently merge runs."""
    match = re.search(r"(\d+)-band", TRAINING_SCHEMA_VERSION)
    assert match, (
        f"{TRAINING_SCHEMA_VERSION!r} does not record a band count; shard "
        "files key on this string to refuse merging incomparable runs")
    assert int(match.group(1)) == len(BANDS), (
        f"schema says {match.group(1)} bands, config ships {len(BANDS)}")


def test_training_populations_are_complete():
    assert BEFORE_TRAINING_TABLE.endswith("/district_train_table")
    assert set(FEATURE_COLLECTIONS) == {
        "forest",
        "water",
        "buildings",
        "barren",
        "agriculture",
    }
    # Class 3 is Barren Land now; both Soil assets and Sand are deliberately gone.
    assert all("soil" not in path for path in FEATURE_COLLECTIONS.values())
    assert all("sand" not in path for path in FEATURE_COLLECTIONS.values())
    assert EXPECTED_ASSET_LABELS == {
        "forest": 0,
        "water": 1,
        "buildings": 2,
        "barren": 3,
        "agriculture": 4,
    }
    assert all("/water_points" not in path for path in FEATURE_COLLECTIONS.values())


def test_season_windows_are_exact():
    assert SEASONS == {
        "winter": {"start": "2025-01-01", "end": "2025-03-01"},
        "summer": {"start": "2025-04-01", "end": "2025-05-30"},
    }


def test_season_windows_are_equal_length():
    """A seasonal accuracy gap is only about season if the windows match.

    Winter used to run 59 days and summer 31, so summer also had roughly half
    the chances of a cloud-free pass. That confounds composite depth with
    phenology, and the winter-to-summer drop could not be attributed to either.
    """
    spans = {
        season: (date.fromisoformat(window["end"])
                 - date.fromisoformat(window["start"])).days
        for season, window in SEASONS.items()
    }
    assert set(spans.values()) == {SEASON_WINDOW_DAYS}
