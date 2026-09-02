import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = sorted((ROOT / "model_training/multi_classification").glob("*_multi.ipynb"))
JABALPUR = sorted(
    (ROOT / "model_testing/multi_classification/jabalpur").glob("*_multi_jabalpur.ipynb")
)
MADHYA_PRADESH = sorted(
    (ROOT / "model_testing/multi_classification/madhya_pradesh").glob("*_multi_mp.ipynb")
)
CONSOLIDATED = (
    ROOT / "model_testing/multi_classification/madhya_pradesh"
    / "seasonal_before_after_all_models.ipynb"
)


def _cells(path):
    return json.loads(path.read_text())["cells"]


def _code(path):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _cells(path)
        if cell["cell_type"] == "code"
    )


def test_ten_model_notebooks_use_canonical_six_class_schema():
    assert len(TRAINING) == 5 and len(JABALPUR) == 5
    for path in TRAINING + JABALPUR:
        source = _code(path)
        assert "from backend.config import" in source, path
        assert "BANDS" in source and "FEATURE_COLLECTIONS" in source, path
        assert "LAND_COVER_CLASSES" in source and "SEASONS" in source, path
        assert "class_names = ['Forest', 'Water', 'Buildings', 'Soil']" not in source, path
        assert "bands = [" not in source, path
        assert "users/cosypix/water_points" not in source, path
        assert "2025-03-01" not in source and "2024-10-01" not in source, path
        assert "PRE_AGRICULTURE_OUTPUTS" in source, path
        # `season` is read by the composite cell, so it has to be bound in the
        # schema cell above it or every one of these notebooks NameErrors on rerun.
        assert 'season = SEASONS["summer"]' in source, path


def test_madhya_pradesh_notebooks_use_canonical_six_class_schema():
    assert len(MADHYA_PRADESH) == 5
    for path in MADHYA_PRADESH:
        source = _code(path)
        assert "from backend.config import" in source, path
        assert "BANDS" in source and "FEATURE_COLLECTIONS" in source, path
        assert "LAND_COVER_CLASSES" in source and "SEASONS" in source, path
        assert "make_classifier(" in source, path
        assert "class_names = ['Forest', 'Water', 'Buildings', 'Soil']" not in source, path
        assert "bands = [" not in source, path
        assert "2024-10-01" not in source, path
        # A four-class matrix embedded in a six-class notebook reads as a result.
        assert all(
            not cell.get("outputs") for cell in _cells(path)
            if cell["cell_type"] == "code"
        ), path


def test_consolidated_notebook_delegates_to_shared_results_and_report():
    source = _code(CONSOLIDATED)
    assert "seasonal_before_after_results.json" in source
    assert "render_markdown" in source
    assert "iter_run_specs" in source
