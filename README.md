# Geo-Engine Terrain Classifier

Four-class land cover classification (Forest, Water, Buildings, Soil) from Sentinel-2 and Sentinel-1 imagery, served as a web dashboard on top of Google Earth Engine.

Draw an area of interest on the map, pick a classifier, and get a classified raster overlay, per-class area figures, and a GeoTIFF export.

## How it works

Models are trained on 4,000 hand-labelled points across Jabalpur district (1,000 per class) and then applied to whatever area the user draws.
The training composite is built from the bounding box of the labelled points themselves, so no point falls outside it.

The feature stack is 27 bands:

| Group | Bands |
|---|---|
| Sentinel-2 reflectance | B2, B3, B4, B8, B11, B12 |
| Spectral indices | NDVI, NDWI, NDBI, SAVI |
| Bareness / built-up indices | BSI, UI, IBI, SWIRratio, BAEI |
| GLCM texture (from B8) | g_contrast, g_ent, g_var, g_idm, g_diss, g_asm |
| Sentinel-1 SAR | VV, VH, VVVH, s_contrast, s_var, s_ent |

Texture and radar are there for one reason.
Bare soil and concrete are nearly identical in Sentinel-2 reflectance, so no combination of the optical bands separates them.
Texture measures local structure rather than brightness, and radar picks up the double-bounce return that buildings produce and bare ground cannot.
Together they took Soil recall across Madhya Pradesh from 7% to 54%.

Five classifiers are selectable: Random Forest (default), SVM with an RBF kernel, Gradient Boosted Trees, CART, and KNN.

## Accuracy, honestly

The accuracy badges in the dashboard (98.7% to 99.35%) describe a random held-out split inside the training area and should not be read as field accuracy.

Measured against ESA WorldCover:

| Scope | Overall |
|---|---|
| Random split inside Jabalpur | 94.3% |
| Spatially blocked split inside Jabalpur | 79.2% to 91.5% |
| Across Madhya Pradesh, 27 features, Random Forest | 73.9% |

The spatially blocked figure is the defensible one.
A random split scores each model against near-duplicates of its own training rows, because labelled points come in clusters and neighbouring 10 m pixels land on both sides of the split.

Known limitations are recorded in [doc/mp_accuracy_report.md](doc/mp_accuracy_report.md).
The short version: arid terrain is still misread as built-up, dry-season deciduous forest is under-detected, and cropland has no valid mapping onto four classes.

## Repository structure

```
backend/                        FastAPI service and the Earth Engine pipeline
  config.py                     Bands, classes, palette, asset paths, model metadata
  gee_classifier.py             Composite building, feature stack, training, classification
  app.py                        HTTP API
frontend/                       Leaflet dashboard (static HTML, CSS, JS)
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
   Both the backend and the notebooks build paths as `{EE_ASSET_ROOT}/jabalpur_forest_points`, `{EE_ASSET_ROOT}/forest_points` and so on, so pointing this at a different account re-homes all of them at once.

4. **Run the dashboard:**
   ```bash
   python run_app.py
   ```
   Serves the API and frontend at `http://127.0.0.1:8000`.

5. **Run the notebooks:**
   Open them from their own directory in Jupyter.
   They read `EE_ASSET_ROOT` from the same `.env`.

## Changes

[doc/CHANGES.md](doc/CHANGES.md) records what changed and why.
