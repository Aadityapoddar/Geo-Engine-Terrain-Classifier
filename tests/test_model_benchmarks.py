import json
from pathlib import Path

import pytest

from backend.config import MODEL_BENCHMARKS, MODEL_METADATA, TRAINING_SCHEMA_VERSION

try:
    from evaluation.artefacts import SPATIAL_UNCERTAINTY, versioned
    RESULTS = versioned(SPATIAL_UNCERTAINTY)
except ImportError:  # production checkout: no evaluation tree
    RESULTS = None


def test_benchmarks_carry_the_current_schema_and_metric():
    for model, metadata in MODEL_METADATA.items():
        benchmark = metadata["benchmark"]
        assert "internal_accuracy" not in metadata
        assert benchmark["training_schema_version"] == TRAINING_SCHEMA_VERSION
        assert (benchmark["metric"]
                == "held_out_district_five_class_overall_accuracy")


def test_dashboard_badges_match_the_measured_test_district_results():
    """The badge is a claim about accuracy, so pin it to the artefact.

    Hard-coding the expected numbers here would only prove that config still
    says what this file says. Reading them back from the results the numbers
    came from is what actually catches a config edited without a rerun, or a
    rerun whose output nobody copied across.
    """
    if RESULTS is None:
        pytest.skip("evaluation/ is not part of this checkout")
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS.name} has not been generated")
    payload = json.loads(RESULTS.read_text())
    for season in ("winter", "summer"):
        pooled = payload["seasons"][season]["test"]["pooled"]
        for model, benchmark in MODEL_BENCHMARKS.items():
            measured = pooled[f"after-{model}"]["overall_accuracy"] * 100
            assert benchmark[season] == pytest.approx(measured, abs=1e-9), (
                f"{model} {season}: config says {benchmark[season]}, "
                f"{RESULTS.name} says {measured}")
