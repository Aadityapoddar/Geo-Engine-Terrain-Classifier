import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app import app
from backend.config import DEFAULT_MODEL


def test_api_recommends_best_validated_default_model():
    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    assert DEFAULT_MODEL == "xgb"
    assert response.json()["default_model"] == DEFAULT_MODEL


def test_frontend_applies_api_default_model():
    source = (REPO / "frontend/js/app.js").read_text()

    assert "selectModel.value = modelsData.default_model" in source
