# Geo-Engine Terrain Classifier

Five-class land use/land cover classification (Vegetation, Water, Built Area, Open Land, Agriculture) from Sentinel-2 and Sentinel-1 imagery, served as a web dashboard on top of Google Earth Engine.

Draw an area of interest on the map, pick a classifier, and get a classified raster overlay, per-class area figures, and a GeoTIFF export.

> **Where the research tooling lives.** This branch carries the deployable
> application only: the backend, the frontend and the tests that guard them.
> The evaluation package, the experiment scripts and the measured artefacts
> this README quotes (`evaluation/`, `scripts/`, `doc/assets/`) live on the
> research branch, and every number below is reproducible from there. Keeping
> them apart is deliberate -- a deploy should not need a statistics stack, and
> a reviewer should not have to guess which files the server actually runs.

## How it works

The current model merges every configured source asset, one per class: Vegetation, Water, Built Area, Open Land and Agriculture, 1,000 labelled points each. The Earth Engine asset ids keep their historical spellings (`jabalpur_forest_points`, `jabalpur_barren_points`) because renaming a remote asset is not a display change -- it breaks any deployment still reading the old id. Sand was dropped in v3 because no public reference product can score it independently. The unrelated legacy `water_points` collection containing 100 points is never used. The preserved `district_train_table` is the before snapshot used for before/after evaluation.
The training composite is built from the bounding box of the labelled points themselves, so no point falls outside it.

Every source asset carries its own `label`, so class identity travels with the points and no caller has to re-assert it. That property is load-bearing: an asset that loses it merges without complaint and then trains a silently empty class. `scripts/audit_training_assets.py` is the check that catches it, and `jabalpur_agriculture_points_labelled` is named for having hit it.

The water points are worth stating plainly, because the wording here previously implied otherwise. `district_train_table` and `jabalpur_water_points` hold the same 1,000 locations: no point differs by more than 6.68 m, which is inside the 7.07 m half-diagonal of a 10 m pixel and is what a `sampleRegions(geometries=True)` round trip does to coordinates. The water class contributes no real difference between the before and after conditions. `scripts/water_point_change.py` measures this.

The classifier uses a 19-band stack, selected by measurement rather than by hand:

| Group | Bands |
|---|---|
| Sentinel-2 reflectance | B4, B8, B11, B12 |
| Core indices | NDVI, NDWI, SAVI |
| Bareness / built-up indices | UI, IBI, SWIRratio |
| GLCM texture (from B8) | g_contrast, g_var, g_idm |
| Sentinel-1 SAR | VV, VH, VVVH, s_contrast, s_var, s_ent |

`B2`, `B3`, `NDBI`, `BSI`, `BAEI`, `g_ent`, `g_diss`, `g_asm` are computed as ingredients of the indices above but are not given to the classifier.

The stack is chosen by `scripts/district_experiments.py --mode ablation` and
`--mode stacks`: a leave-one-band-out sweep over all 27 bands, then five
candidate stacks scored on the 15 development districts and once on the 29
held-out test districts (`scripts/select_band_stack.py`,
`scripts/build_stack_comparison.py`, `scripts/adopt_band_stack.py`). Ranking is
by development accuracy averaged over both reported seasons; the seasons
disagree on their own, so a single-season rule would be arbitrary.

An earlier 19-band cut, chosen on the same pooled points it was reported
against, turns out to be the worst of the five candidates. The current stack
has the same band count and shares only 13 of them.

Hyperparameters are tuned the same way, on development districts only
(`--mode grid`, then `scripts/adopt_tuned_params.py`), and live in exactly one
place -- `MODEL_METADATA` in `backend/config.py`, which `make_classifier` reads.

Texture and radar are there for one reason.
Bare soil and concrete are nearly identical in Sentinel-2 reflectance, so no combination of the optical bands separates them.
Texture measures local structure rather than brightness, and radar picks up the double-bounce return that buildings produce and bare ground cannot.
Together they take open-land recall from 42% to 73% in winter and 33% to 55% in summer, measured on the development districts against the five-class consensus (`scripts/district_experiments.py --mode ablation`). Removing the radar family alone costs 5.7 points of overall accuracy in winter and 6.5 in summer -- more than any other feature family by a factor of three.

An earlier version of this figure (7% to 54%) came from a four-class WorldCover tile study on a composite belonging to neither reported season, and is superseded. Open land remains the weakest class: nothing in the training set labels the bright dry terrain of Malwa, Chambal and Bundelkhand, and open-land recall falls about 3 points per 100 km of distance from the labelled area while agriculture recall rises by a similar amount.

Five classifiers are selectable: Random Forest (default), SVM with an RBF kernel, Smile Gradient Tree Boosting (GTB), CART, and KNN.

## Accuracy, honestly

The legacy accuracy badges in the dashboard (98.7% to 99.35%) describe an older random held-out split and should not be read as field accuracy.

The seasonal evaluator compares all five models for Winter (`2025-01-01` to exclusive `2025-03-01`) and Summer (`2025-04-01` to exclusive `2025-05-30`). Both windows are 59 days: they used to be 59 and 31, which meant the shorter one also had half the chances of a cloud-free pass, so composite depth was confounded with season. `scripts/composite_depth_audit.py` measures what is left of that -- under the matched windows the mean per-district scene counts differ by about two Sentinel-2 granules and not at all in Sentinel-1. External statewide evaluation uses high-confidence agreement among WorldCover, Dynamic World, WorldCereal, GHSL, and OPERA DSWx. The project's own labelled points are used for training only; scoring happens exclusively against that public-map consensus across Madhya Pradesh, with bare-ground project classes collapsing to the reference's Bare class. Generated results live in `doc/assets/seasonal_before_after_results.json`, and [the before/after report](doc/before_after_accuracy_report.md) covers all five models.

Read the like-for-like table in that report rather than the raw held-out accuracy. Before is scored on the four classes it could actually predict, After on six, so raw overall accuracy falls for every model even where nothing regressed.

Measured against ESA WorldCover:

| Scope | Overall |
|---|---|
| Random split inside Jabalpur | 94.3% |
| Spatially blocked split inside Jabalpur | 79.2% to 91.5% |
| Across Madhya Pradesh, 27 features, Random Forest | 73.9% |

The spatially blocked figure is the defensible one.
A random split scores each model against near-duplicates of its own training rows, because labelled points come in clusters and neighbouring 10 m pixels land on both sides of the split.
`scripts/label_provenance_audit.py` puts a number on how tightly: the median nearest-neighbour distance between two Open Land points is 29 m, and 984 of the 1,000 sit within 100 m of another one.

### The district split

Districts are partitioned once, by spatial block, before anything is selected (`evaluation/splits.py`):

| Half | Districts | What it is used for |
|---|---|---|
| Training | Jabalpur, Katni, Dindori, Mandla | Nothing. Withheld: each physically contains labelled training points. |
| Development | 15 | Feature-stack, classifier and hyperparameter choice. |
| Test | 29 | Reported once. Never consulted before that. |

The four withheld districts are not a formality. The training set is described everywhere as 5,000 points from Jabalpur; 872 of them are not, and they land in Katni, Dindori and Mandla, which were simultaneously being scored as independent test districts. A 100 m buffer keeps individual test points off individual training points, but it does not make a district independent of a model trained inside it.

Confidence intervals come from resampling whole districts rather than points (`scripts/spatial_uncertainty.py`), because reference samples inside one district share landscape, phenology and reference-map error. Model comparisons are paired per-district differences, not a ranking of point estimates -- on the held-out districts the top three classifiers are not separable.

Overall accuracy is also reported area-adjusted (`scripts/area_adjusted_accuracy.py`). The sample takes 20 points per class per district, so the unweighted matrix describes a landscape that is one fifth water; the Olofsson et al. stratified estimator reweights each (district, class) stratum by its measured share of the mapped area.

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

4. **Export the Open Land points (one time):**
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
