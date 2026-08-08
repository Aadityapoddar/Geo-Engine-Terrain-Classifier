# Geo-Engine Terrain Classifier

Five-class land cover classification (Forest, Water, Buildings, Barren Land, Agriculture) from Sentinel-2 and Sentinel-1 imagery, served as a web dashboard on top of Google Earth Engine.

Draw an area of interest on the map, pick a classifier, and get a classified raster overlay, per-class area figures, and a GeoTIFF export.

## How it works

The current model merges every configured source asset: Jabalpur Forest, Jabalpur Water (1,000 points), Buildings, Barren Land, and Agriculture label 4. Sand was dropped in v3 because no public reference product can score it independently. The unrelated legacy `water_points` collection containing 100 points is never used. The preserved `district_train_table` is the before snapshot used for before/after evaluation.
The training composite is built from the bounding box of the labelled points themselves, so no point falls outside it.

Every source asset carries its own `label`, so class identity travels with the points and no caller has to re-assert it. That property is load-bearing: an asset that loses it merges without complaint and then trains a silently empty class. `scripts/audit_training_assets.py` is the check that catches it, and `jabalpur_agriculture_points_labelled` is named for having hit it.

The water points are worth stating plainly, because the wording here previously implied otherwise. `district_train_table` and `jabalpur_water_points` hold the same 1,000 locations: no point differs by more than 6.68 m, which is inside the 7.07 m half-diagonal of a 10 m pixel and is what a `sampleRegions(geometries=True)` round trip does to coordinates. The water class contributes no real difference between the before and after conditions. `scripts/water_point_change.py` measures this.

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
Together they took bare-ground recall across Madhya Pradesh from 7% to 54%.

That 54% figure was measured with `soil_points_mp`, 1,000 statewide bare-ground labels, which has since been dropped in favour of Jabalpur-only Barren Land points. Expect statewide bare ground to regress accordingly: nothing now labels the bright dry terrain of Malwa, Chambal and Bundelkhand, which is the terrain the model reads as built-up.

Five classifiers are selectable: Random Forest (default), SVM with an RBF kernel, Gradient Boosted Trees, CART, and KNN.

## Accuracy, honestly

The legacy accuracy badges in the dashboard (98.7% to 99.35%) describe an older random held-out split and should not be read as field accuracy.

The seasonal evaluator compares all five models for Winter (`2025-01-01` to exclusive `2025-02-28`) and Summer (`2025-03-31` to exclusive `2025-04-30`). External statewide evaluation uses high-confidence agreement among WorldCover, Dynamic World, WorldCereal, GHSL, and OPERA DSWx. The project's own labelled points are used for training only; scoring happens exclusively against that public-map consensus across Madhya Pradesh, with bare-ground project classes collapsing to the reference's Bare class. Generated results live in `doc/assets/seasonal_before_after_results.json`, and [the before/after report](doc/before_after_accuracy_report.md) covers all five models.

Read the like-for-like table in that report rather than the raw held-out accuracy. Before is scored on the four classes it could actually predict, After on six, so raw overall accuracy falls for every model even where nothing regressed.

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

4. **Export the Barren Land points (one time):**
   Class 3 loads from the `jabalpur_barren_points` asset, which starts life as a
   drawn Geometry Import inside the `jabalpur_barren_land_points` Code Editor
   script and therefore is not reachable from a checkout.
   Paste [scripts/export_barren_land_asset.js](scripts/export_barren_land_asset.js)
   at the bottom of that script, run it, and start the task from the Tasks tab.
   It sets `label: 3` explicitly, which is the whole point: a Geometry Import
   exports with no attributes, and an unlabelled asset merges into training
   without complaint and then contributes nothing.
   Confirm it landed with:
   ```bash
   python scripts/audit_training_assets.py --output doc/assets/training_asset_audit.json
   ```

5. **Run the dashboard:**
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

6. **Run the notebooks:**
   Open them from their own directory in Jupyter.
   They read `EE_ASSET_ROOT` from the same `.env`.

## Changes

[doc/CHANGES.md](doc/CHANGES.md) records what changed and why.
