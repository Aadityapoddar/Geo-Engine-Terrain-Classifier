# Geo-Engine Terrain Classifier

Six-class land cover classification (Forest, Water, Buildings, Soil, Sand, Agriculture) from Sentinel-2 and Sentinel-1 imagery, served as a web dashboard on top of Google Earth Engine.

Draw an area of interest on the map, pick a classifier, and get a classified raster overlay, per-class area figures, and a GeoTIFF export.

## How it works

The current model merges every configured source asset: Jabalpur Forest, repositioned Jabalpur Water (1,000 points), Buildings, Soil, MP Soil, MP Sand, and Agriculture label 5. The unrelated legacy `water_points` collection containing 100 points is never used. The preserved `district_train_table` provides the original 1,000 Jabalpur water geometries for before/after evaluation.
The training composite is built from the bounding box of the labelled points themselves, so no point falls outside it.

The classifier uses the validated 19-band cut:

| Group | Bands |
|---|---|
| Sentinel-2 reflectance | B3, B11, B12 |
| Bareness / built-up indices | BSI, UI, IBI, SWIRratio, BAEI |
| GLCM texture (from B8) | g_contrast, g_ent, g_var, g_idm, g_diss, g_asm |
| Sentinel-1 SAR | VV, VH, VVVH, s_contrast, s_var |

`B2`, `B4`, `B8`, `NDVI`, `NDWI`, `NDBI`, `SAVI`, and `s_ent` remain available for feature construction where needed but are excluded from classifier input.

Texture and radar are there for one reason.
Bare soil and concrete are nearly identical in Sentinel-2 reflectance, so no combination of the optical bands separates them.
Texture measures local structure rather than brightness, and radar picks up the double-bounce return that buildings produce and bare ground cannot.
Together they took Soil recall across Madhya Pradesh from 7% to 54%.

Five classifiers are selectable: Random Forest (default), SVM with an RBF kernel, Gradient Boosted Trees, CART, and KNN.

## Accuracy, honestly

The legacy accuracy badges in the dashboard (98.7% to 99.35%) describe an older random held-out split and should not be read as field accuracy.

The seasonal evaluator compares all five models for Winter (`2025-01-01` to exclusive `2025-02-28`) and Summer (`2025-03-31` to exclusive `2025-04-30`). External statewide evaluation uses high-confidence agreement among WorldCover, Dynamic World, WorldCereal, GHSL, and OPERA DSWx. Soil and Sand collapse to Bare for this five-class public-reference comparison; separate six-class scores use spatially held-out project labels. Generated results live in `doc/assets/seasonal_before_after_results.json` and [the seasonal report](doc/seasonal_before_after_accuracy.md) once all Earth Engine batch exports complete.

Measured against ESA WorldCover:

| Scope | Overall |
|---|---|
| Random split inside Jabalpur | 94.3% |
| Spatially blocked split inside Jabalpur | 79.2% to 91.5% |
| Across Madhya Pradesh, 27 features, Random Forest | 73.9% |

The spatially blocked figure is the defensible one.
A random split scores each model against near-duplicates of its own training rows, because labelled points come in clusters and neighbouring 10 m pixels land on both sides of the split.

Historical limitations are recorded in [doc/mp_accuracy_report.md](doc/mp_accuracy_report.md).
The short version: arid terrain is still misread as built-up, dry-season deciduous forest is under-detected, and cropland has no valid mapping onto four classes.

## Repository structure

```
backend/                        FastAPI service and the Earth Engine pipeline
  config.py                     Bands, classes, palette, asset paths, model metadata
  gee_classifier.py             Composite building, feature stack, training, classification
  app.py                        HTTP API
frontend/                       v1 Leaflet dashboard (static HTML, CSS, JS)
frontend-next/                  v2 dashboard: Vite + TypeScript + MapLibre GL
feature_engineering/            Cloud-cover, seasonal and grid-search sensitivity studies
  binary_classification/
  multi_classification/
model_training/                 Notebooks that fit each classifier
  multi_classification/
model_testing/                  Notebooks that evaluate each classifier
  multi_classification/
    jabalpur/                   Evaluation inside the labelled area
    madhya_pradesh/             Evaluation across the wider state
doc/                            Accuracy study, change log, generated figures
tests/                          API tests
run_app.py                      Launches the backend and opens the dashboard
```

Notebooks write their figures to `doc/assets/` via a relative path, so run them from their own directory.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Authenticate with Earth Engine:**
   ```bash
   earthengine authenticate
   ```

3. **Configure `.env`:**
   ```env
   EE_PROJECT_ID=your-project-id
   EE_ASSET_ROOT=users/your-ee-username
   ```
   `EE_ASSET_ROOT` is the folder every training-point FeatureCollection is loaded from.
   Both the backend and notebooks derive all current asset paths from this root.

4. **Run the dashboard:**
   ```bash
   python run_app.py
   ```
   Serves the API and the v1 frontend at `http://127.0.0.1:8000`.

   For the v2 frontend, leave that running and start Vite alongside it:
   ```bash
   cd frontend-next && npm install && npm run dev
   ```
   It proxies `/api` to the backend and serves the dashboard at `http://127.0.0.1:5173`.
   [frontend-next/README.md](frontend-next/README.md) explains what it does differently, and [doc/frontend_performance_report.md](doc/frontend_performance_report.md) has the measured before/after.

5. **Run the notebooks:**
   Open them from their own directory in Jupyter.
   They read `EE_ASSET_ROOT` from the same `.env`.

## Changes

[doc/CHANGES.md](doc/CHANGES.md) records what changed and why.
