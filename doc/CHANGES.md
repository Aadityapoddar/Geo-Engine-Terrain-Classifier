# Changes

A plain record of every change made on the `feature/sar-texture-features-and-mp-evaluation` branch, and why.

This file is maintained by hand.
It is not a generated changelog.

## Summary

The branch does four things.
It adds texture and radar features to fix the soil-versus-built-up confusion that made the dashboard unusable outside Jabalpur.
It corrects three bugs that made parts of the product unreachable or its accuracy figures misleading.
It reorganises the notebook tree so folder names describe their contents.
It documents the measured accuracy, including where the models fail.

## Backend

### Feature stack expanded from 10 bands to 27

`backend/config.py` (`BANDS`), `backend/gee_classifier.py` (`_add_spectral_indices`, new `_add_sar`).

Added, in three groups:

- **Bareness and built-up indices**: BSI, UI, IBI, SWIRratio, BAEI.
- **GLCM texture from B8**: g_contrast, g_ent, g_var, g_idm, g_diss, g_asm, quantised to 32 grey levels over a 3-pixel window.
- **Sentinel-1 SAR**: VV, VH, VVVH plus s_contrast, s_var, s_ent texture, from `COPERNICUS/S1_GRD`.

The problem being solved is that bare soil and concrete are nearly identical in Sentinel-2 reflectance.
Measured signatures over Rajasthan desert are red 0.222 against built-up 0.226, and SWIR 0.334 against 0.342.
Nothing in the original ten bands separates them, which is why the dashboard reported 68% Buildings over open sand and why Soil recall across Madhya Pradesh sat at 8%.

Each group was measured separately against the ten-band baseline on 1,274 MP points with Random Forest:

| Feature set | Overall | Soil recall | Buildings precision |
|---|---|---|---|
| Baseline, 10 bands | 58.6% | 7% | 49% |
| + bareness indices | 60.4% | 11% | 51% |
| + GLCM texture | 65.3% | 37% | 58% |
| + Sentinel-1 SAR | 70.6% | 50% | 59% |
| All of the above | 73.9% | 54% | 63% |

Texture fixes precision because built-up is structurally rough where bare soil is smooth, which is information no single pixel carries.
Radar fixes recall because buildings produce a double-bounce return that bare ground cannot make at any brightness.
They address opposite halves of the same confusion, which is why using both beats either alone.

SAR values are rescaled from dB into roughly 0 to 1 before use.
Left in raw dB the -25 to 5 range would dominate the distance-based RBF kernel that the SVM model uses.

**What this costs.**
A second satellite dependency now exists, with orbit, incidence-angle and speckle effects, and an untested date-widening fallback if no Sentinel-1 scenes fall in the requested window.
Classification is slower.
Arid terrain got worse rather than better: Thar dunes now read 80.9% Buildings under Random Forest, against 52% under the pre-radar SVM configuration, because no bright arid soil is labelled in the training set.

### Fixed: training composite was built over the wrong geometry

`backend/gee_classifier.py`, `get_trained_classifier`.

The training composite was built over a fixed campus polygon (`CAMPUS_GEOJSON`), roughly 2 km across.
The training labels had already moved to 4,000 points spanning Jabalpur district.
`sampleRegions` silently returns nothing for points outside the composite, so most of the training set was being discarded with no error.

The training geometry is now derived from the bounds of the labelled points themselves, so it always covers them.

### Fixed: the KNN model was unreachable

`backend/gee_classifier.py`.

`ee.Classifier.smileKnn` does not exist.
The Earth Engine Python API spells it `smileKNN`.
Selecting KNN in the dashboard raised `AttributeError` before any Earth Engine call was made.

### Fixed: training assets were hardcoded to one account

`backend/config.py`, `FEATURE_COLLECTIONS`.

Paths were hardcoded to `users/cosypix/...` and pointed at the older campus-only point sets.
They now resolve from a new `EE_ASSET_ROOT` environment variable and point at the district-wide `jabalpur_*_points` collections, matching what the notebooks load.

### Removed: `CAMPUS_GEOJSON`

`backend/config.py`.

With training geometry now derived from the labels, nothing read this constant.
Its comment ("the classifier is ALWAYS trained on this area") had also become false.
`CAMPUS_MAP_CENTER`, which the UI uses to zoom to the study area, is untouched.

### Renamed for accuracy

`campus_collection` and `campus_composite` became `train_collection` and `train_composite`, and the surrounding docstring and comments no longer claim training happens on the campus.

## Notebooks

### `model_testing/multi_classification/jabalpur/` split into two folders

The folder held both the Jabalpur notebooks and the Madhya Pradesh ones, so its name described only half its contents.
The five `*_multi_mp.ipynb` notebooks moved to a sibling `madhya_pradesh/` folder.

Both folders sit at the same depth, so the `../../../doc/assets/` figure paths inside them resolve unchanged.
This was verified: no file anywhere in the repository referenced either folder by path.

### `feature_engineering/` gained per-scope subfolders

`binary_classification/` and `multi_classification/` were added, mirroring the layout of `model_testing/`.
`gridsearch_tuning.ipynb` moved under `binary_classification/`.
Four binary and three multi-class sensitivity notebooks are now tracked that were previously untracked.

### Fixed: feature-engineering notebooks could not save figures

Those notebooks called `savefig('fe/....png')`.
No `fe/` directory exists in this repository, so every save raised `FileNotFoundError`, which is visible in the notebooks' own stored outputs.
Others wrote bare filenames that landed beside the notebook rather than in the assets folder.

All of them now write to `../../doc/assets/`, matching the convention already used in `model_testing/`, with an `os.makedirs` guard ahead of the first save.

### Random train/test split replaced with a spatial block split

`model_testing/multi_classification/*/*.ipynb`.

The previous `randomColumn` split divided pixels, not places.
Labelled points come in clusters, so neighbouring 10 m pixels off the same lake or forest patch landed on both sides of the split, and each model was scored against near-duplicates of its own training rows.

Points are now binned into roughly 11 km cells by coordinate, and whole cells go to either train or test.
Reported accuracy drops as a result: 94.3% under the random split against 79.2% to 91.5% blocked.
The lower figure is the honest one, and the gap between them is spatial autocorrelation.

### Fixed: one notebook overwrote another's figure

`rf_multi_jabalpur.ipynb` was titled "CART Multi-Class Confusion Matrix" and wrote `multi_cart_jabalpur_cm.png`, overwriting the CART figure with the Random Forest one.
All ten notebooks were checked; each title and figure name now matches its own model and region.

### Asset paths resolved from the environment

The notebooks loaded FeatureCollections from a hardcoded Earth Engine account, so nobody else could run them.
They now read `EE_ASSET_ROOT` from `.env`, matching the backend.

### Removed `.gitkeep` placeholders

`model_testing/multi_classification/jabalpur/.gitkeep` is no longer needed now that the folder holds real notebooks.
`model_testing/binary_classification/double_campus_area/` and `model_testing/binary_classification/jabalpur/` were reserving directories that were never filled, so those two paths leave the repository entirely.

## Repository configuration

### Fixed: the frontend entry point was being ignored

`.gitignore`.

A blanket `*.html` rule, meant for exported reports, also matched `frontend/index.html`, the application's only entry point.
It was never committed, so a fresh clone had a backend that served a 404 at `/`.
The rule is now negated for `frontend/`, and `index.html` is tracked.

`.DS_Store` was also added to the ignore list.

## Documentation

- `doc/mp_accuracy_report.md` is new.
  It evaluates all five classifiers against ESA WorldCover across Madhya Pradesh, and is the source for every accuracy figure quoted above.
  It records the failures as well as the wins, including the arid-terrain regression and the fact that all five models agree on 71% of points while being correct only 76% of the time when they do.
- Eleven confusion-matrix figures added under `doc/assets/`, covering all five models for both Jabalpur and MP, plus a before/after comparison.
- `README.md` was rewritten.
  It described a binary forest-versus-non-forest campus study and pointed at four directories (`fe/`, `nb/`, `rf/`, `cal_area/`) that do not exist in this repository.
  It now describes the four-class product, the 27-band feature stack, the real directory layout, and the measured accuracy rather than the dashboard badges.
- This file was added.

## Known issues not addressed here

Carried over from `doc/mp_accuracy_report.md`, in rough order of impact:

1. Arid and rocky terrain is still read as built-up, and radar made it worse.
2. Soil recall is 54% outside Jabalpur against 95% inside it.
3. Buildings precision is 63%, so roughly a third of reported built-up is not built-up.
4. The SVM model regressed on Water precision, from 79% to 63%, because its `gamma=1.0` was tuned for 10 features rather than 27.
   Jabalpur city centre reports 38.8% water under SVM.
   The default stays on Random Forest until SVM is retuned.
5. Forest recall is 68%, because dry deciduous forest is leaf-off during the default March to April window.
6. Cropland has no valid mapping onto four classes, and it is the largest land type in Madhya Pradesh.
7. The dashboard accuracy badges still advertise 98.7% to 99.35%, which matches nothing measured here.
