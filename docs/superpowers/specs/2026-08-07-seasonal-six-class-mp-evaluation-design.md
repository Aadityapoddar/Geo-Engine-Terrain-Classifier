# Seasonal Six-Class Madhya Pradesh Evaluation Design

Date: 2026-08-07

## Goal

Add Agriculture as label `5`, train every supported classifier with every applicable training asset, compare the old and revised Jabalpur water labels in winter and summer, validate the models across Madhya Pradesh with independent public reference datasets, and expose the resulting six-class model through the backend and frontend.

## Fixed class mapping

| Label | Class |
|---:|---|
| 0 | Forest |
| 1 | Water |
| 2 | Buildings |
| 3 | Soil |
| 4 | Sand |
| 5 | Agriculture |

Existing Soil and Sand labels remain unchanged. The Agriculture export must set `label=5` on every feature.

## Training populations

### Before snapshot

The before condition uses the preserved `district_train_table` asset. It contains the original 1,000 Jabalpur water-point geometries before repositioning and the complete five-class training population available at that time:

- Forest: 1,000
- Water: 1,000
- Buildings: 1,000
- Soil: 2,000, combining Jabalpur and MP soil assets
- Sand: 500

The unrelated legacy `water_points` asset containing 100 points is never used.

### After assets

The after condition reads every current source asset rather than a hand-selected subset:

- `jabalpur_forest_points`
- `jabalpur_water_points` (the repositioned 1,000-point collection)
- `jabalpur_building_points`
- `jabalpur_soil_points`
- `soil_points_mp`
- `sand_points_mp_labelled`
- `jabalpur_agriculture_points`

The Agriculture collection is exported from the saved Earth Engine Code Editor geometry import. Before training, every collection is validated for existence, nonzero size, expected label, point geometry, non-null label values, and usable Sentinel samples. The report records raw and post-sampling counts.

## Seasons and feature stack

The two Earth Engine `filterDate` windows are used exactly as supplied. Earth Engine treats the end date as exclusive.

| Season | Start | End |
|---|---|---|
| Winter | 2025-01-01 | 2025-02-28 |
| Summer | 2025-03-31 | 2025-04-30 |

Models use the verified 19-band cut from `doc/assets/mp_band_cut_confusion.json`. The eight excluded classifier inputs are `B2`, `B4`, `B8`, `NDVI`, `NDWI`, `NDBI`, `SAVI`, and `s_ent`.

The retained classifier inputs are:

`B3`, `B11`, `B12`, `BSI`, `UI`, `IBI`, `SWIRratio`, `BAEI`, `g_contrast`, `g_ent`, `g_var`, `g_idm`, `g_diss`, `g_asm`, `VV`, `VH`, `VVVH`, `s_contrast`, `s_var`.

Raw bands excluded from classifier input remain available internally when required to calculate retained indices and textures.

## Models

The five existing Earth Engine classifiers and their current parameters remain unchanged:

- Random Forest
- RBF SVM
- Gradient Tree Boost, exposed as XGB
- CART
- KNN

The primary matrix contains 20 conditions: 2 seasons x 2 training populations x 5 models. Fixed seeds and the existing deterministic spatially blocked split are reused for held-out project-label evaluation.

## Madhya Pradesh reference design

No single public map supplies accurate six-class truth. Public products also merge Soil and Sand into a bare-ground class. Therefore the report separates two claims:

1. **Five-class external accuracy:** Forest, Water, Buildings, Agriculture, and Bare, where model predictions `Soil` and `Sand` are collapsed to `Bare`.
2. **Six-class held-out accuracy:** all six project labels, evaluated only against spatially held-out project points.

This avoids presenting a guessed Sand reference as ground truth.

### Reference sources

- ESA WorldCover v200 at 10 m supplies stable Tree, Cropland, Built-up, Bare/sparse, and Permanent water labels.
- Dynamic World supplies season-matched 10 m probabilities and labels. A candidate must have the target class as top label with probability at least 0.70.
- ESA WorldCereal active-cropland markers support the Agriculture reference, using the applicable AEZ seasonal product.
- GHSL 10 m built surface supports Buildings; a high-confidence candidate requires at least 50 square metres of built surface in its 10 m cell.
- OPERA DSWx-HLS supports season-matched Water; open or partial surface-water classes are accepted after cloud and no-data exclusions.

A reference candidate must agree between WorldCover and season-matched Dynamic World, and must also agree with the relevant class-specific product where one exists. Bare candidates exclude pixels marked as crop, built, or water by the specialist products. Conflicts and low-confidence pixels are masked rather than force-labelled.

### Statewide sampling

The MP boundary and its districts come from the existing GAUL-based boundary logic. For each season and external class, sample up to 20 high-confidence points per district at 10 m, using a fixed seed and geometries. This produces statewide spatial coverage of up to roughly 1,000 points per class while avoiding dominance by large or easy districts. The same exported seasonal evaluation table is reused by every model and both before/after conditions.

The evaluation samples are external to all project training assets. A spatial exclusion buffer around every training point prevents leakage.

## Metrics and report

For each of the 20 conditions, record:

- overall accuracy
- Cohen's kappa
- macro precision, recall, and F1
- per-class precision and recall
- confusion matrix
- training and evaluation counts
- district coverage
- missing or masked reference counts

The before/after table reports absolute and percentage-point deltas separately for Winter and Summer. It includes both the five-class external metric and the six-class held-out metric. Confusion matrices are generated with a fixed label order. The report explicitly describes temporal mismatch in WorldCover, WorldCereal, and GHSL and does not call external maps perfect ground truth.

Deliverables:

- a machine-readable JSON result file
- a Markdown report under `doc/`
- confusion-matrix images under `doc/assets/`
- updated notebook outputs using the same result generator

## Code and notebook architecture

A shared evaluation module owns season definitions, band definitions, model construction, asset validation, reference construction, sampling, metrics, and result serialization. A deterministic runner performs Earth Engine exports and evaluations without copying model logic across notebooks.

The five training notebooks and five Jabalpur evaluation notebooks are synchronized to the six-class mapping and 19-band inputs. A consolidated seasonal MP before/after notebook calls the shared runner and renders the final report tables and matrices. MP notebooks without Agriculture ground truth are not silently reinterpreted as six-class truth.

The backend uses the same shared class mapping, 19-band list, model parameters, and complete after-asset list. Its visualization range and area summaries are derived from configuration rather than hard-coded class counts. Cache identity includes the training schema version so a stale five-class classifier cannot survive the update.

Both frontends remain API-driven. Six-class legends, colors, charts, area cards, and exports render Agriculture without hard-coded four- or five-class assumptions. The backend-served v1 frontend is the required runtime deliverable; v2 is updated where its dynamic rendering still has assumptions.

## Failure handling

- Stop before training if an asset is missing, empty, has an unexpected label, or produces zero valid 19-band samples.
- Export Agriculture to a new asset ID; do not overwrite unrelated assets.
- Submit long Earth Engine table jobs in bounded batches and record task IDs and terminal status. Restricted project quota is reported as an external blocker, not hidden by partial results.
- Never substitute `water_points` for either 1,000-point Jabalpur water population.
- Preserve old metrics and artifacts; write new before/after results to distinct files.
- Label missing external reference coverage as unavailable rather than zero accuracy.

## Verification

1. Unit tests assert six ordered classes, the exact 19-band list, complete after-asset membership, dynamic visualization bounds, and six class-area entries.
2. Asset audit verifies counts, labels, geometry types, bounds, and non-null sampled bands.
3. Evaluation-table audit verifies class balance, district coverage, spatial exclusion from training points, and deterministic seeds.
4. All 20 conditions complete and serialize the required metrics with fixed confusion-matrix order.
5. Backend tests and the full Python test suite pass.
6. The v2 TypeScript build passes when affected.
7. The backend-served frontend starts locally, returns a healthy API response, and is visually checked for Agriculture legend/chart/overlay behavior.
8. Report totals are cross-checked against the JSON source before delivery.

## Scope boundary

This work measures model agreement with high-confidence external references across MP; it does not claim field-survey accuracy. Sand remains separately measurable only with held-out project labels until an independent sand-specific reference is available.
