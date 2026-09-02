# Changes

## 2026-09-02 - The feature stack and the hyperparameters, re-selected

The band study this project shipped on was a four-class WorldCover experiment
over twelve rectangular tiles, using a March-April composite belonging to
neither reported season and a pooled evaluation set that also chose the stack.
Every number it produced was quoted beside results that share none of those
things. It has been re-run from scratch under the district protocol, and the
conclusion inverts.

### The shipped stack was the worst of the five candidates

Leave-one-band-out over all 27 bands, Smile GTB, scored district by district on
the 15 development districts. Then five candidate stacks, chosen on development
and scored once on the 29 held-out test districts:

| Stack | Bands | Mean development OA |
|---|---|---|
| `drop_summer` | 19 | 81.43% |
| `drop_winter` | 25 | 81.06% |
| `all27` | 27 | 80.71% |
| `b22` | 22 | 80.03% |
| `b19` (was shipped) | 19 | 78.01% |

The winner has the same number of bands as the retired cut and shares only
13 of them. It keeps B4, B8, NDVI, NDWI, SAVI, s_ent,
which the old cut discarded, and drops B3, BSI, BAEI, g_ent, g_diss, g_asm
instead. On the test districts that is worth +1.48 points in winter and +4.50
in summer.

Almost nothing in the stack is redundant. Winter can spare two bands of
twenty-seven, summer eight, and no band is spare in both -- so the earlier
"fewer bands is better" reading was never about dimensionality. It was about a
cut chosen on the points it was then reported against.

### The seasons disagree, so the selection rule averages them

Winter development prefers a 25-band stack and summer a different 19-band one.
Selecting on one season would have been arbitrary for a paper that reports two,
so stacks are ranked by development accuracy averaged over both. The rule lives
in one function that both the adoption and its test call, after an earlier
version had it in two places that drifted apart.

### Hyperparameters retuned on the stack that won

A grid over all five classifiers, development districts only:

| Model | Was | Now |
|---|---|---|
| Random Forest | 200 trees, bag 0.3, maxNodes 10 | 300 trees, bag 0.5, no node cap |
| SVM | gamma 1.0, C 100 | gamma 0.1, C 10 |
| Smile GTB | 100 trees | 300 trees |
| KNN | k=5 | k=9 |
| CART | maxNodes 20 | unchanged |

Random Forest's inherited `maxNodes=10` was the binding constraint the audit
suspected: ten split points however many features the forest is offered. The
search removes the cap.

### Supporting changes

- `make_classifier` now reads its settings from `MODEL_METADATA` instead of
carrying a second copy. Retuning edits one place, so the served description,
the trained model and the paper cannot disagree.
- `BAND_STACKS["b19"]` was an alias for `BANDS`. Adopting a new stack therefore
redefined a historical reference point in the comparison table; the named
stacks are now literals and the shipped one is carried separately.
- `scripts/district_experiments.py` and the arbiter audit take
`--worker-count`/`--worker-index`, striping districts the way the statewide
runner does. The development sweep went from roughly forty-five minutes to two.
- Workers no longer summarise their own slice. A quarter of the districts
reported as if it were all of them is a number that looks plausible and is
wrong; `merge_experiment_shards.py` does it once the slices are joined, and
refuses to merge across schema versions or modes.
- `evaluation/artefacts.py` names the current results version once. It had been
spelled into five files, which had already produced one figure built from a
different run than the table beside it.

## 2026-09-01 - Method corrections from the journal-readiness audit

An external audit of the manuscript raised twenty issues, nine of which are
about the code rather than the writing. This is what changed and what it cost.

### The evaluation set was not external

`scripts/label_provenance_audit.py` re-measured the five training assets
against the district polygons the evaluation uses. 872 of the 5,000 "Jabalpur"
points are not in Jabalpur: 868 in Katni, three in Dindori, one in Mandla, all
within 23 km of the boundary. Katni was separated from Jabalpur in 1998, so a
wider historical outline explains the placement.

Those three districts were being scored as independent test districts while
holding training data. The 100 m leakage buffer keeps individual test points off
individual training points; it does nothing about a district the model was
partly trained inside. All four are now withheld (`evaluation/splits.py`).

The same audit found the labels are far more clustered than the paper claimed.
Median nearest-neighbour distance within a class runs from 29 m (Barren Land) to
371 m (Buildings), and 984 of the 1,000 Barren Land points sit within 100 m of
another Barren Land point. At 10 m pixels that is three pixels apart, which is
the whole explanation for the 94.3% random-split accuracy.

### Selection and reporting used the same districts

The 27-to-22-to-19 band cut, the classifier ranking and the reported accuracy
all came off one statewide population, so the headline number was a maximum over
choices rather than an estimate. The districts are now partitioned once, by
one-degree spatial block, before anything is selected: 15 development, 29 test,
4 withheld. Every choice is made on development; test is scored once.

This is not cosmetic. On the held-out districts the top three classifiers stop
being separable - the Smile GTB advantage over SVM is inside its confidence
interval in both seasons.

### The seasonal windows were not comparable

Winter ran 59 days and summer 31, so summer also had roughly half the chances of
a cloud-free pass, and composite depth was confounded with season. Both windows
are now 59 days: winter 1 January-28 February, summer 1 April-29 May.
`scripts/composite_depth_audit.py` measures what is left - the two seasons now
differ by about two Sentinel-2 granules per district and not at all in
Sentinel-1, and optical coverage is complete in all 48 districts in both.

### Sentinel-1 mixed orbit directions and substituted out-of-season imagery

- The collection now filters `orbitProperties_pass = DESCENDING`. Ascending and
descending passes see the same slope or wall from opposite sides, so a median
over both added a look-direction term to every radar feature. The paper already
claimed descending; the code did not do it.
- The fallback to the 2024-2025 archive is gone from the experimental path. A
2024 scene standing in for an April 2025 one describes a different ground state,
and the seasonal comparison would have been partly a comparison of fallback
rates. It survives only in `classify_and_analyze`, the dashboard path, where a
user who draws a polygon wants a map rather than a blank and no accuracy is
claimed. Measured across all 96 district-seasons, no window is empty, so the
experimental path never needed it.
- The VV/VH ratio was already correct - a difference of decibels is the log of
the linear-power ratio - but nothing said so, and the rescaling that follows it
looked like it might be part of the ratio. Now documented at the call site.

### The model is not XGBoost

`ee.Classifier.smileGradientTreeBoost` is Smile's gradient tree boosting. It is
not the XGBoost of Chen and Guestrin and has neither the regularised objective
nor the sparsity-aware split finding. The model id is now `gtb` throughout, with
`xgb` still accepted by the API and `make_classifier` so old links keep working.

### Accuracy had no uncertainty and was not area-adjusted

- `scripts/spatial_uncertainty.py` resamples whole districts rather than points,
because reference samples inside one district share landscape, phenology and
reference-map error. It reports 95% intervals for OA, kappa, macro F1 and
per-class F1, and compares classifiers by paired per-district differences.
- `scripts/area_adjusted_accuracy.py` implements the Olofsson et al. stratified
estimator over (district, reference class) strata, weighted by measured area
from `scripts/reference_stratum_weights.py`. The sample takes 20 points per
class per district, so the unweighted matrix describes a landscape that is one
fifth water.

### The reference was assumed correct

`scripts/consensus_arbiter_audit.py` scores the five-product consensus against
ESRI's 10 m Annual Land Cover, which contributes nothing to it and whose 2024
epoch is the closest to the 2025 composites. Where they disagree this cannot say
which is wrong, but it bounds how much of the reported model error is really
reference error - previously assumed to be zero.

### Schema

`TRAINING_SCHEMA_VERSION` is `five-class-19-band-v4-matched-windows`. The class
inventory and band cut are unchanged from v3, but the composite underneath them
is not, so v3 and v4 results are not comparable and the shard files refuse to
merge across the boundary.

## 2026-08-13 - The terrain overlay is rendered at the scale the classifier was trained at

- The map and the numbers beside it disagreed by 6x on Water.
On Jabalpur district the analytics panel reported Water at 5.65% of the AOI while the overlay painted it across 36.53% of the classified pixels.
The numbers were the trustworthy half.
- Cause: `_static_overlay` capped the render at 2048 px, which for a district is ~60 m per pixel.
Earth Engine re-evaluates the entire chain - composite, indices, GLCM texture, SAR, classify - at whatever scale it is asked for, so the overlay was not a downsampled picture of the 10 m classification.
It was a different classification.
- Two of the nineteen features are GLCM textures over a 3x3 *pixel* window.
That window is 30 m on the training grid and 180 m on a 60 m grid, computed over mean-pyramided reflectance, so the smooth low-texture signature that separates water from everything else showed up across most of the district.
- The same request, reduced at four scales, Water as a share of the AOI:

| scale | 10 m | 30 m | 50 m | 100 m |
|---|---|---|---|---|
| Water | 5.65% | 31.99% | 28.54% | 26.52% |

- This was not a palette or blending artefact.
Every pixel in the old PNG came back an exact palette colour, so nothing was interpolated - the model had genuinely predicted those classes at that scale.
- The overlay is now rendered at 10 ground metres and nowhere else.
Earth Engine will not do that in a single request - a district on a 10 m grid is ~12600x7100 px and it answers `Reprojection output too large` - so `render_overlay_png` cuts the AOI into 1792 px tiles, renders each at 10 m, and stitches them here.
- Tiles are cut on the Web Mercator pixel grid rather than on latitude, so they butt together without a seam, and each tile's composite is built on a 24 px padded region because GLCM and `focalMode` read a neighbourhood and a composite cut to the tile edge misclassifies every boundary.
- Web Mercator metres are inflated by 1/cos(lat), so asking for "10 m" in EPSG:3857 buys ~9.2 ground metres at Jabalpur.
`_render_grid` picks the mercator scale that lands on 10 ground metres instead, so the render sits on the training grid rather than near it.
- Cost is real: ~2 min per tile, and Earth Engine returns 429 above three concurrent renders, so a district is ~25 minutes.
It runs on a background thread, `/api/classify` returns the statistics immediately, and the frontend polls `/api/overlay/{key}`.
The finished PNG is cached on disk under `.overlay_cache/`, keyed by geometry, model, dates, cloud threshold, smoothing and `TRAINING_SCHEMA_VERSION`, so re-running an AOI is instant and panning and zooming still costs one flat image.
- `OVERLAY_RENDER_WORKERS` drops the concurrency below three.
A project in Restricted Mode shares one Earth Engine budget across everything, so a render at full tilt will starve incoming classify requests.
- Verified on a 6 km AOI end to end: reported against painted, per class, is within 0.9 points everywhere, against a 6x error on Water before.
The residual is the difference between the Mercator render grid and the reducer's native projection, not a difference in model output.

## 2026-08-12 - Feature bands rescaled to a common range

- The GLCM texture bands were appended raw while every other band sat in 0-1 or -1..1.
Measured on the winter training samples, `g_var` alone was 51% of the squared Euclidean distance between feature vectors and the four texture moments together were 97%, leaving the entire optical block and the three deliberately-scaled radar bands to share the remaining 3%.
- `FEATURE_RANGES` in `backend/gee_classifier.py` maps the nine unbounded or wide-range bands onto 0-1 with the same `unitScale().clamp()` pattern the SAR amplitude bands already used: the six GLCM moments plus `s_ent`, and the `SWIRratio` and `BAEI` ratios, which had no upper bound at all and could spike wherever their denominators approached zero.
- Bounds are the winter p99 of the labelled training samples.
They must come from that class-stratified sample and not from a district-wide pixel sweep.
Districts are overwhelmingly vegetation and cropland, so a district percentile badly understates a built-up index on the class-balanced set the classifier actually sees: `BAEI` measured p99 2.7 across districts against 8.1 across the training samples.
The first attempt used the district figure and set the `BAEI` bound to 3.0, which clamped 12.3% of winter rows to a single tied value - cutting into the body of the distribution rather than trimming a tail.
- Bounds set too wide are the opposite failure and are equally real, but silent: they compress a band into a fraction of the scale and under-weight it in the same distance metric the table exists to balance.
The first attempt had `g_contrast` at 50.0 against a true p99 of 20.85, and no check caught it because nothing saturated.
- Verify any bound change with the saturation check: every band should sit near 1% saturated, neither 12% nor 0%.
Current bounds put all nine between 0.00% and 0.90% in both seasons.
- After the change the four texture moments account for 25% of the distance and no single band exceeds 14%.
- Only the two distance-based classifiers were mis-served, and both improve sharply on the whole-MP external benchmark: SVM winter 71.34% to 84.10% (Kappa 0.642 to 0.801) and KNN winter 73.68% to 83.16% (Kappa 0.671 to 0.790).
Summer gains are +8.56 and +3.91 points.
- The three tree models are threshold-based and therefore scale-invariant.
Random Forest and CART are bit-identical to the published figures across every run, and XGBoost moves 0.13 points.
All five were re-run rather than assumed unchanged, because the clamp does flatten the top of each band into a tie and that could in principle remove a split.
- `libsvm` keeps `gamma=1.0`, which was compensating for the broken scale and is a reasonable value for 19 unit-scaled features; it is the next knob to try, not part of this change.
- Whole-MP results are re-aggregated into `doc/assets/mp_external_results_v5.json` from 96/96 district-season shards, and `MODEL_BENCHMARKS` reads from it.
`DEFAULT_MODEL` stays `xgb`, which still leads in both seasons, though SVM has closed from 13.7 points behind in winter to 1.0.
- `doc/paper/five_class_terrain_mp.tex` is updated: the abstract, the aggregate accuracy table, both per-class F1 tables with recomputed bold entries, the discussion, the limitations and the conclusion.
The paper previously attributed SVM's last-place ranking to an untuned RBF `gamma`, and that attribution is now stated as incorrect and replaced with the scaling explanation.
The paragraph on SVM over-calling Water is kept as a record of the symptom, since Water precision rising from 0.61 to 0.92 is the clearest evidence that the defect was in the feature representation rather than the kernel.

## 2026-08-12 - Band ablation could not measure anything

- `scripts/band_ablation.py` imported `BANDS` from `backend.config`, and that list was cut from 27 entries to 19 by this very study.
Every name in `STACKS` therefore named a band that no longer existed, all three stacks resolved to the same 19 bands, and the script silently reproduced its own published numbers instead of measuring anything.
The 27-band stack is now pinned in the script with assertions that fail loudly if it drifts from `GROUPS` or stops being a superset of the production bands.
- The script also carried a copy of the Sentinel-1 band maths rather than calling the shipped code, and the copy missed the texture rescaling when it was added.
`add_sar_bands` in `backend/gee_classifier.py` is now shared by both callers.
The copy existed to avoid `_add_sar`'s per-tile `.getInfo()` probes, which the split preserves.
- The MP evaluation table is exported as twelve per-tile shards and merged, because compositing all twelve inside one export exceeds the Earth Engine batch compute limit and dies with "Computation timed out".
- `socket.setdefaulttimeout(600)` is set, matching `run_sharded_mp_external.py`.
Without it a dropped connection blocks `getInfo` forever: the process stays alive, stops logging, and never finishes, which no liveness check can detect.
- Known improvement, not taken: the exported tables store post-rescale values, so any future bound change forces a full re-export through a batch queue that runs roughly one task per hour.
Exporting raw values and applying `FEATURE_RANGES` at load time would make bound changes free.
- The rebuilt ablation confirms the 19-band cut on every classifier, not only the tree ensembles.
Random Forest gains 4.5 points over the full 27-band stack, Gradient Boosting 1.0 and SVM 1.2, against a seed standard error below 0.19.
The published table recorded the cut as neutral for SVM, which was an artefact of the unscaled features rather than a property of the cut.
- The held-out design still confirms it: 80.50% overall and 0.740 Kappa for the 19-band cut on eastern MP against 77.42% and 0.699 for the full stack, a margin that widens from 1.20 to 3.08 points.
- Unresolved: Random Forest scores 77.33% on the 19-band cut against a published 75.28%, and the gap does not close when the BAEI bound is corrected.
Saturation was the obvious candidate and is now ruled out, since BAEI saturates 0.00% of the rebuilt evaluation table and the figure is unchanged from the run that saturated 13.5%.
The remaining difference between the two runs is the evaluation table itself, which was re-sampled, so the cause is not established and no causal claim is made in the paper.
- The forward-additive attribution in `doc/mp_accuracy_report.md`, source of the "bareness indices contributed 1.8 points" claim, is a separate experiment that has not been re-run and still reflects pre-rescaling features.

## 2026-08-08 — Sand dropped; five-class v3 inventory

- Sand (`sand_points_mp_labelled`, previously label 4) is no longer trained.
No public reference product distinguishes river sand from bare ground, so the class could never be independently tested, and it competed with Barren Land for the same spectral space.
- Agriculture moves from label 5 to label 4 so the inventory stays contiguous: Forest 0, Water 1, Buildings 2, Barren Land 3, Agriculture 4.
`jabalpur_agriculture_points_labelled` was deleted and re-exported from `jabalpur_agriculture_points_updated` with the new label; `scripts/export_agriculture_asset.py` now reads the label from `EXPECTED_ASSET_LABELS` so it cannot drift from the backend inventory.
- `TRAINING_SCHEMA_VERSION` is now `five-class-19-band-v3-no-sand`.
- The external five-class collapse is now per condition, because Before and After no longer share a label inventory.
Before (the preserved soil-era `district_train_table`) collapses Soil 3 and Sand 4 to Bare; After collapses only Barren Land 3 to Bare and maps Agriculture 4 to the reference Agriculture class.
`PROJECT_TO_REFERENCE_LABEL` in `evaluation/references.py` carries both maps, and every evaluation script passes the condition through.
- Evaluation policy stated explicitly: project-created assets are used for training only.
Scoring happens exclusively against the public-map consensus (WorldCover, Dynamic World, WorldCereal, GHSL, OPERA DSWx) across Madhya Pradesh districts, with reference samples buffered at least 100 m from any training point.
- The unstamped whole-MP shard files under `doc/assets/mp_external_shards_worker_*.json` predate schema stamping, describe an older inventory, and are rejected by the aggregator; the statewide external evaluation restarts under v3.

## 2026-08-07 — Barren Land replaces Soil as class 3

- Class 3 is now `Barren Land`, sourced from `jabalpur_barren_land_points` (1,000 points).
- Both Soil assets are dropped: `jabalpur_soil_points` and `soil_points_mp`.
- Label numbering is unchanged, so Sand stays 4 and Agriculture stays 5 and neither asset needs re-exporting.
- **This is a knowing regression for statewide bare ground.** `soil_points_mp` was 1,000 points spread across Madhya Pradesh and is the measured reason bare-ground recall outside Jabalpur went from 7% to 54%; the ablation table further down this file is the evidence. Barren Land is Jabalpur-only, so the arid terrain of Malwa, Chambal and Bundelkhand now has no labelled bare ground and is expected to read as built-up again. This trade was requested explicitly.
- `scripts/export_barren_land_asset.js` exports the drawn `barren_points` geometry with an explicit `label`, because a Geometry Import exports with no attributes at all. That is not hypothetical: it is exactly how the Agriculture class ended up training on nothing.
- The before/after accuracy report was generated before this swap and describes the Soil schema. It is accurate for what it measured.
- `TRAINING_SCHEMA_VERSION` is now `six-class-19-band-v2-barren`, up from `six-class-19-band-v1`.
This is the fix for a trap the swap exposed: the version keys the trained-classifier cache and is stamped into every results artefact, but it had not changed when class 3 changed identity, so a Soil-schema confusion matrix and a Barren-schema one were indistinguishable and would sum without complaint into a single report whose rows meant two different things.
The sharded runner now stamps the schema into its checkpoint and refuses to resume a file from another one; the aggregator and the results merger both reject mismatched or unstamped input.
- The 37 completed whole-Madhya-Pradesh district shards in `doc/assets/mp_external_shards_worker_*.json` were computed under v1 and are now correctly rejected by the aggregator. The statewide external evaluation has to start over under v2 once the Barren Land asset exists.

## 2026-08-07 — Agriculture actually reaches training

The Agriculture class was configured everywhere and trained nowhere.

- `jabalpur_agriculture_points_updated` held 1,000 points with no attributes at all: every feature had empty properties.
Training samples with `sampleRegions(properties=['label'])` and drops null labels, so all 1,000 points were discarded in silence and the six-class model trained on five classes.
Confirmed through the live API: an area of interest drawn on the Agriculture training points returned 0.00% Agriculture.
- Exported the labelled collection as `jabalpur_agriculture_points_labelled` via `scripts/export_agriculture_asset.py`, following the `sand_points_mp_labelled` precedent rather than overwriting the original import.
Soil holds label 3 and Sand holds 4, so Agriculture takes 5; a feature arriving with label 4 is remapped to 5 on export.
Training population goes from 5,500 samples with zero Agriculture to 6,500 with 1,000.
- Removed the `_inline_agriculture` GeoJSON workaround that the evaluation scripts used to route around the unlabelled asset.
- Verified end to end through the real backend path, on the same area of interest, with only the Agriculture label fix changed between the two runs:

  | Class | Before | After |
  |---|---:|---:|
  | Forest | 33.31% | 4.65% |
  | Water | 6.00% | 4.17% |
  | Buildings | 18.20% | 18.64% |
  | Bare ground (class 3) | 42.49% | 16.68% |
  | Sand | 0.00% | 0.00% |
  | Agriculture | **0.00%** | **55.86%** |

  Farmland that the model had been splitting between bare ground and Forest is now read as Agriculture, which is what the area actually is. Evidence in `doc/assets/frontend_class_mix_before.json` and `frontend_class_mix_after.json`.
- The first export attempt built the collection from 1,000 inline features and sat `PENDING` for over two hours under the project's restricted compute quota. Relabelling the geometries already in Earth Engine server-side keeps the export graph small and succeeded; `scripts/export_agriculture_asset.py --source-asset` is that path.
- The five Madhya Pradesh notebooks were still four-class on a hand-built ten-band composite. They now share `backend.config` and take their classifier parameters from `make_classifier`, so a notebook cannot drift from the served model. Their stale four-class outputs are cleared.
- Fixed a `NameError` in all ten previously updated notebooks: the composite cell read `season` before the schema cell bound it, so every one of them crashed on rerun.
- Measured the water points instead of assuming. Before and After hold the same 1,000 locations: no point moved further than 6.68 m, under the 7.07 m half-diagonal of a 10 m pixel, which is the signature of a `sampleRegions(geometries=True)` round trip rather than an edit. The water class contributes no real before/after difference.
- Added a like-for-like held-out comparison. Raw overall accuracy drops from Before to After for every model because After is scored on six classes where Before could only attempt four; restricting to the shared classes is the comparison that answers whether anything regressed.

## 2026-08-07 — six classes, seasonal evaluation, 19-band cut

- Added Agriculture label 5; retained Forest 0, Water 1, Buildings 2, Soil 3, Sand 4.
- Current training merges all seven configured assets. `jabalpur_water_points` is the repositioned 1,000-point collection; legacy `water_points` (100) is excluded. The preserved `district_train_table` remains the before snapshot with original 1,000 water geometries.
- Reduced classifier input from 27 to the validated 19 bands. Excluded `B2`, `B4`, `B8`, `NDVI`, `NDWI`, `NDBI`, `SAVI`, and `s_ent`.
- Fixed Winter and Summer windows at `2025-01-01`–`2025-02-28` and `2025-03-31`–`2025-04-30`; end dates are exclusive.
- Added a checkpointed 20-run evaluator: five models × before/after × Winter/Summer. External MP reference uses WorldCover + Dynamic World consensus with WorldCereal, GHSL, and OPERA DSWx refinements; Soil and Sand collapse to Bare only for external metrics.
- Updated ten model notebooks plus a consolidated seasonal report notebook.
- Added `/api/config`; both frontends read canonical dates and render API-supplied class rows/colors, including Agriculture.
- Earth Engine batch IDs and final measured accuracies are recorded in the seasonal manifest/report; queued or quota-blocked work is not reported as completed accuracy.

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
