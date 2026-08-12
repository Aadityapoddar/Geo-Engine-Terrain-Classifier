# Geo-Engine Terrain Classifier - self-contained app

Sentinel-2 + Sentinel-1 land cover classification on Google Earth Engine, served as a web app.
Everything needed to run it is in this folder: the API backend, the web frontend, and the notebooks.

## Setup

Requires Python 3.10+ and a Google account with Earth Engine access to project `earth-engine-484907`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate
python run_app.py
```

That opens http://127.0.0.1:8000 in your browser.
Draw a rectangle on the map, pick a model, and classify.

## Configuration

Copy `.env.example` to `.env`. Every asset lives in the Earth Engine project named by `EE_PROJECT_ID`, and each class points at one of them by exact name:

| Variable | Class | Label |
| --- | --- | --- |
| `EE_ASSET_FOREST` | Forest | 0 |
| `EE_ASSET_WATER` | Water | 1 |
| `EE_ASSET_BUILDINGS` | Buildings | 2 |
| `EE_ASSET_BARREN` | Barren Land | 3 |
| `EE_ASSET_AGRICULTURE` | Agriculture | 4 |

Renaming an asset in the project is a one-line edit here, not a code change.

Each asset must be a `FeatureCollection` of points carrying a numeric `label` matching the table above.
An asset with no `label` attribute is dropped silently at sampling time and the model trains without that class, so check that first if a class comes back empty.

## What is here

| Path | Contents |
| --- | --- |
| `backend/` | FastAPI app, Earth Engine classifier, class and model config |
| `frontend/` | Map UI, served by the backend at `/` |
| `model_training/` | Notebooks that train each classifier |
| `model_testing/` | Notebooks that score them over Jabalpur and Madhya Pradesh |
| `feature_engineering/` | Notebooks behind the band, season, and cloud-threshold choices |

Run notebooks from this folder, not from inside their subdirectories, so they can import `backend` and read `.env`:

```bash
source .venv/bin/activate
pip install jupyter
jupyter lab
```

## Models

Five classifiers over the same 19-band feature space (spectral + indices + GLCM texture + Sentinel-1 radar).
XGBoost is the default: 85.1% winter and 70.3% summer agreement with the five-class public-map consensus across 48 Madhya Pradesh districts.

## Hosted deployment

Set `EE_SERVICE_ACCOUNT_KEY` to the full service-account JSON instead of running `earthengine authenticate`.
