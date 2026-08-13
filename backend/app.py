import os
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from backend.config import (
        DEFAULT_MAP_CENTER,
        CAMPUS_MAP_CENTER,
        LAND_COVER_CLASSES,
        MODEL_METADATA,
        DEFAULT_MODEL,
        BANDS,
        SEASONS,
        TRAINING_SCHEMA_VERSION,
    )
    from backend.gee_classifier import (
        init_ee,
        classify_and_analyze,
        overlay_status,
        OVERLAY_CACHE_DIR,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from backend.config import (
        DEFAULT_MAP_CENTER,
        CAMPUS_MAP_CENTER,
        LAND_COVER_CLASSES,
        MODEL_METADATA,
        DEFAULT_MODEL,
        BANDS,
        SEASONS,
        TRAINING_SCHEMA_VERSION,
    )
    from backend.gee_classifier import (
        init_ee,
        classify_and_analyze,
        overlay_status,
        OVERLAY_CACHE_DIR,
    )

app = FastAPI(
    title="Geo-Engine Terrain Classifier API",
    description="Backend API for Sentinel-2 land cover classification using Google Earth Engine and Machine Learning",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    geometry: Dict[str, Any] = Field(..., description="GeoJSON Geometry (Polygon or Rectangle) marked on map")
    model_type: str = Field(DEFAULT_MODEL, description="Classifier model ID: 'rf', 'svm', 'xgb', 'cart', 'knn'")
    start_date: str = Field(SEASONS["summer"]["start"], description="Sentinel-2 composite start date (YYYY-MM-DD)")
    end_date: str = Field(SEASONS["summer"]["end"], description="Sentinel-2 composite end date (YYYY-MM-DD, exclusive)")
    cloud_threshold: float = Field(15.0, description="Max cloud cover percentage filter (1-50%)")
    smoothing: bool = Field(True, description="Enable 1px focal mode spatial smoothing")


@app.on_event("startup")
def startup_event():
    try:
        init_ee()
    except Exception as e:
        print(f"Startup Warning: Earth Engine initialization pending: {e}")


@app.get("/api/health")
def get_health():
    ee_status = False
    try:
        ee_status = init_ee()
    except Exception:
        ee_status = False

    return {
        "status": "online",
        "gee_initialized": ee_status,
        "default_map_center": DEFAULT_MAP_CENTER,
        "campus_map_center": CAMPUS_MAP_CENTER
    }


@app.get("/api/models")
def get_models():
    return {
        "default_model": DEFAULT_MODEL,
        "models": MODEL_METADATA,
        "classes": LAND_COVER_CLASSES,
        "map_centers": {
            "default": DEFAULT_MAP_CENTER,
            "campus": CAMPUS_MAP_CENTER
        }
    }


@app.get("/api/config")
def get_config():
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "classes": LAND_COVER_CLASSES,
        "bands": BANDS,
        "seasons": SEASONS,
        "default_season": "summer",
    }


@app.post("/api/classify")
def run_classification(payload: ClassifyRequest):
    if not payload.geometry or "coordinates" not in payload.geometry:
        raise HTTPException(status_code=400, detail="Invalid GeoJSON geometry. Must contain valid 'coordinates'.")

    valid_models = ["rf", "svm", "xgb", "cart", "knn"]
    if payload.model_type.lower() not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type '{payload.model_type}'. Choose from {valid_models}")

    try:
        results = classify_and_analyze(
            geometry_dict=payload.geometry,
            model_type=payload.model_type,
            start_date=payload.start_date,
            end_date=payload.end_date,
            cloud_threshold=payload.cloud_threshold,
            smoothing=payload.smoothing
        )
        return results
    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Earth Engine classification failed: {str(e)}")


@app.get("/api/overlay/{key}")
def get_overlay_status(key: str):
    """How far the AOI's terrain render has got.

    The overlay is rendered at the classifier's 10 m training scale, which for a
    district is minutes of Earth Engine compute -- far longer than a request can
    wait -- so `/api/classify` returns the statistics immediately and the
    frontend polls this until the PNG lands in the cache.
    """
    return overlay_status(key)


@app.post("/api/export")
def export_report(payload: ClassifyRequest):
    try:
        results = classify_and_analyze(
            geometry_dict=payload.geometry,
            model_type=payload.model_type,
            start_date=payload.start_date,
            end_date=payload.end_date,
            cloud_threshold=payload.cloud_threshold,
            smoothing=payload.smoothing
        )

        geojson_report = {
            "type": "Feature",
            "geometry": payload.geometry,
            "properties": {
                "model": results["model"],
                "summary": results["summary"],
                "individual_class_areas": results["individual_class_areas"],
                "geotiff_download_url": results["export"].get("geotiff_url", "")
            }
        }
        return JSONResponse(content=geojson_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export generation failed: {str(e)}")


# Rendered terrain overlays. Cached on disk and served flat, so panning and
# zooming costs the client one HTTP request and Earth Engine nothing.
os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)
app.mount("/overlays", StaticFiles(directory=OVERLAY_CACHE_DIR), name="overlays")

# Serve static frontend files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
