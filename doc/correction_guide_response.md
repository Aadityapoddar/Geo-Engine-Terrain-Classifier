# Response to the constructive correction guide of 5 September 2026

This records what was changed in response to `LULC_05Sep2026_Constructive_Correction_Guide.docx`, item by item, and what was deliberately not changed.
Every claim below is backed by a file in `doc/assets/result_archive/` or by a named script.

The revised manuscript is `new.tex` / `new.pdf`.

## 1. The result archive comes first

The guide's first demand was that the manuscript stop quoting numbers no single stored record produces.

`scripts/build_result_archive.py` now writes `doc/assets/result_archive/`:

| File | Contents |
| --- | --- |
| `predictions.csv` | One row per sampled reference pixel: coordinates, stratum, reference label, and one predicted label per model. |
| `reproduction_check.csv` | The stored evaluation against an independent re-run of the same sample, district by district and model by model. |
| `experiment_manifest.csv` | Run identifier, training season, feature stack, hyperparameters, evaluation population, selection objective, whether the run influenced a choice. |
| `district_split.csv` | Every GAUL ADM2 unit with role, block identifier, centroid, per-season sample count, district area and consensus-covered area. |
| `district_confusion_matrices.csv` | The full 5x5 matrix per district, season and model. |
| `metrics.csv` | Pooled OA, Kappa, macro F1 and per-class precision, recall and F1 for every run. |
| `stratum_areas.csv` | Eligible area and sample count per (season, district, consensus class), with an exclusion reason where a stratum has area but no sample. |
| `reference_domain_coverage.csv` | District area, covered area and both agreement estimates with the uncertainty method. |
| `paired_contrasts.csv` | All eight paired comparisons with interval, raw and Holm-adjusted p, tie and zero counts, and the test variant used. |
| `protocol_comparison.csv` | Every model under both split protocols in both seasons, with training size, fold size and fold class composition. |
| `reference_inventory.csv` | Each reference product with asset id, version, epoch, native grid, specialist condition and remap. |
| `feature_experiments.csv` | Every cumulative, leave-one-band-out, leave-one-family-out and candidate-stack run, with its source file and whether it predates the hyperparameter search. |
| `spatial_transfer.csv` | Per-district distance and accuracy, with the distance regression fitted on all 44 evaluation districts and on the 29 test districts separately. |

The rebuild asserts, for every matrix it prints a metric from, that `n = sum(C)`, `OA = trace(C)/n`, and that each per-class precision, recall and F1 is the corresponding ratio of counts.
Superseded records are kept rather than overwritten: the split-protocol experiment exists as both `jabalpur_split_protocols.json` and `jabalpur_split_protocols_v2.json`.

Two of these files are records of different kinds, and it took building them to see that the distinction mattered.
The manuscript reports the **frozen evaluation**, and that does not change: re-running a held-out evaluation after its results have been read, then reporting the re-run, is how a held-out set stops being held out whatever the intention.
`predictions.csv` is an **audit record** drawn afterwards, by rebuilding the same seeded sample and classifying it again — 9,281 rows over all 96 district-seasons, with one predicted label per model.

It does not reproduce the stored run exactly, and the pattern of the difference turned out to be a finding in its own right rather than noise.
Across the 768 district-model matrices, 535 reference points out of 74,248 changed class.
They are not spread evenly:

| Classifier | Matrices reproducing exactly | Points moved |
| --- | --- | --- |
| Random Forest, SVM, KNN, CART | 95 of 96 each | 1 each, all in winter Umaria |
| Smile GTB (selected stack) | 41 of 96 | 116 |
| Smile GTB (three alternative stacks) | 33-57 of 96 | 106-163 |

The composites, the consensus construction and the seeded stratified sample therefore reproduce essentially perfectly — one pixel in 74,248, and the same pixel for all four deterministic classifiers.
The instability is Smile GTB's own: `samplingRate` 0.7 with seed 0 does not give run-to-run determinism in Earth Engine's implementation.

That matters for the paper's central claim.
Between the two runs the selected model's pooled test accuracy moves from 87.41% to 87.24% in winter and from 73.85% to 74.37% in summer.
Its winter margins over SVM (+1.03) and Random Forest (+0.97) are the same order as the variation it shows against itself.
So there are now two independent reasons not to read a winter ranking off these numbers — the Holm-adjusted p values of 0.125, and the fact that the margin is not comfortably larger than the leading model's own reproducibility band.
The manuscript states this in the results, the limitations, the abstract and the conclusion, and `reproduction_check.csv` gives the comparison district by district.

### The 86.67 / 86.93 recheck

The guide asked for this specifically, and there turn out to be three 27-band development figures, not two, now all in `feature_experiments.csv` with their sources:

| Figure | Winter | Summer | Source | Tuning |
| --- | --- | --- | --- | --- |
| Cumulative sweep, Table XI last row and the leave-one-band-out baseline | 86.67 | 74.75 | `band_ablation_v4.json` | pre-search |
| Leave-one-family-out baseline, Fig. 5 | 86.73 | 74.33 | `band_family_ablation_v4.json` | pre-search |
| Candidate stack `all27`, Table XIII | 86.93 | 73.98 | frozen run | post-search |

The manuscript previously described the gap as a constant 0.26 points.
It is 0.26 in winter and 0.77 in the other direction in summer, so tuning is not a uniform improvement, and the two sweeps also have slightly different baselines because they are separate runs.
Both facts are now stated where the numbers appear, and Fig. 5's caption names its own baseline so its deltas are read within the panel.

## 2. Fig. 1 regenerated, not relabelled

`scripts/rebuild_jabalpur_plate.py` rebuilds the plate from the delivered raster.

The published version carried three unsupportable claims.
It named the classifier "Gradient Boosted Trees (XGBoost), 100 trees" when the model is Earth Engine's Smile gradient tree boosting with 300 trees, shrinkage 0.1 and a 10-node cap.
It printed 556,920 ha of mapped area against a district polygon whose measured geodesic area is 4,021.3 km2 (402,133 ha), because class areas were counted as pixels times a nominal 100 m2 rather than measured with `ee.Image.pixelArea`.
And it put the 29-district external accuracy in the corner of a single-district map with nothing to say the two describe different ground.

The rebuilt plate takes its AOI from the GAUL ADM2 feature by code (ADM2 17730), measures its geodesic area, sums `pixelArea` by class under the same mask that produced the map, reports the raw and the majority-filtered raster separately, states the analytical grid and scale apart from the Web Mercator display projection, reads its classifier field from `backend.config`, and carries no accuracy figure at all.

Two things about the measurement are worth separating, and both turned out to matter more than expected.

`pixelArea` returns each pixel's true ground area, so the **total** is correct on whatever grid the reduction runs on.
The 43,840,650 classified pixels cover 4,011.65 km2 against the polygon's geodesic 4,021.33.
Counting them at a nominal 100 m2 instead gives 4,384 km2 — high by exactly the 1/cos(23.2 deg) a degree grid implies, which is the arithmetic behind the published plate's inflated total.

The class **shares** are not scale-invariant, and the size of the effect is the reason this took three attempts.
The same classified image reduced at 200 m, 100 m, 30 m and 10 m gives built-area totals of 1,321, 1,207, 1,048 and 571 km2.
A 30 m accounting of a 10 m map overstates built area by 83%: a scattered small-patch class is preferentially captured by a coarse grid when the class surrounding it is not.

Getting to 10 m needed the district sharded into 91 tiles.
A single whole-district reduction at 10 m does not return — not grouped, not as masked per-class sums, not as a frequency histogram, with or without `tileScale` — because evaluating a 19-band classifier carrying two GLCM stacks over 44 million pixels exceeds what one interactive Earth Engine request will do.
That is a limit on the request, not on the computation, and it is the same limit this project already answers by sharding its statewide evaluation into 96 jobs.
The tiled sum is exact: it returns the same 4,011.65 km2 as the whole-district `pixelArea` measurement, so no area is lost or double-counted at tile boundaries.

An earlier draft of this document said the current model puts built area "near a quarter" of the district and inferred that the published plate must have come from a different model.
That was an artefact of the 30 m reduction and is withdrawn.
At 10 m built area is 14.24%, against the published plate's 411.8 km2, which is 10.3% of the true district area.
A real difference remains, but it is much smaller than the coarse reduction suggested, and the published plate's dominant error was its area arithmetic rather than necessarily a different model.

The 3x3 majority filter does not change the picture: built area moves -0.7%, vegetation -0.3%, open land +0.6%, and only water changes materially at -10.2%, isolated bright pixels being what a majority filter removes well.
So the delivered dashboard product inherits the raw raster's built-area behaviour, and it carries no measured agreement figure of its own.

## 3. The estimation subsection that was missing

Section IV-F of the manuscript now defines the covered reference domain, the stratum, the weight, the estimator and its variance explicitly, and states what it does not estimate.

The covered domain is 107,461 km2 of the test districts' 190,330 km2 in winter (56.5%) and 104,753 km2 (55.0%) in summer.
Area-weighted agreement is 75.50% +/- 3.32 in winter and 58.69% +/- 4.01 in summer, against unweighted 87.41% and 73.85%.
The interval is the design-based Olofsson variance with a normal 95% half-width, with no finite-population correction, and it is distinguished from the district bootstrap used elsewhere, which answers a different question.

Two accounting details are reported rather than absorbed.
Two of the 138 summer strata have nonzero area and no sampled point; their weight is not silently redistributed, and the estimate is renormalised onto the 99.9996% of the summer domain that sampled strata represent.
Three summer test points fall in a stratum with no measured consensus area at the 30 m weighting scale and carry no weight, so the summer weighted estimate rests on 2,689 of the 2,692 points the unweighted figure uses.

## 4. The split map and the covered domain

`scripts/build_split_map.py` produces Fig. 4: the four withheld districts, the 15 development districts, the 29 test districts, the labelled points at their own coordinates, and a second panel shading each district by the fraction the winter consensus labels at all.
`district_split.csv` carries the same information as identifiers.

Plotting the points surfaced a countable fact the manuscript had not stated: 5,000 labelled features occupy 4,998 distinct coordinates (two exact duplicates) and 4,988 distinct cells on the 10 m sampling grid.

## 5. Fig. 9 and Table 13 rebuilt together

The embedded CART matrices implied n = 1,545 at 89.26% and n = 1,676 at 76.43%, against a table reporting 1,509 and 1,616 at 88.5% and 84.3%.
Nothing on disk could adjudicate, because `scripts/jabalpur_split_protocols.py` discarded the confusion matrices it had already computed.

It now stores them, along with the training size and the fold's class composition, and the experiment was re-run for both seasons under the frozen production configuration (`jabalpur_split_protocols_v2.json`).
`scripts/build_protocol_figure.py` generates the figure and the table from that one record.

The re-run also exposed the confound the old comparison hid.
The two protocols do not hold out the same mixture of classes: the random fold inherits the training set's 1,000-per-class balance (296/309/320/294/290), while the blocked fold assigns whole one-degree cells and lands 35% open land (232/157/213/564/450), and it trains on 3,384 points against 3,491.
The figure therefore reports class-balanced accuracy beside overall accuracy.
The protocol effect survives standardisation and changes size: the winter gap is 3.41 points of overall accuracy and 4.69 class-balanced; the summer gap is 12.37 and 9.93.

## 6. The seasonal comparison, measured rather than asserted

`scripts/seasonal_support_audit.py` measures what the manuscript had argued away.

Per-pixel Sentinel-2 observation depth, counted after the scene filter and the QA60 mask, averages 17.5 in winter against 15.4 in summer, with the fifth percentile falling from 7-10 to 7-9.
Summer rests on about 12% fewer usable optical observations per pixel, which the matched granule counts could not show.
Sentinel-1 depth is identical between seasons.

Clipping runs the opposite way to the obvious worry.
Optical GLCM contrast, the most clipped band, is clipped in 7.3% of winter pixels against 1.7% of summer pixels; SAR contrast 2.7% against 2.0%; everything else under 1% in both.
Shared winter scaling is therefore not a mechanism that could explain the summer drop.

The pure-phenology conclusion is removed.
The seasonal difference is now attributed to seasonal spectral ambiguity, a smaller and differently composed summer reference sample, and a modest reduction in optical observation depth, with the statement that the design does not isolate their contributions.

The seasonal branch is also settled: `scripts/run_sharded_mp_external.py` fits a separate model per season on the same labelled points, feature list and frozen hyperparameters.
Summer is not frozen winter weights, and the manuscript now says so.

## 7. The distance regression, refitted on held-out ground

The guide asked for the held-out-district estimate and the slope interval to be reported separately rather than folded into the pooled fit.

Doing so changed what there is to say.
The pooled 44-district winter slope is +0.55 points per 100 km (95% CI -0.81 to +1.90, p = 0.43).
On the 29 test districts alone it is +1.49 (95% CI -0.35 to +3.32, p = 0.12), and the summer test slope is +2.42 (95% CI +0.12 to +4.73, p = 0.049).
Every fit has a positive sign: what little association exists runs towards higher agreement further from the labelled area, which is not what a transfer argument predicts.
The manuscript reports both fits, notes that the summer test result is a single unadjusted test chosen after the pooled analysis was seen, and does not lean on it.

## 8. Paired comparisons and multiplicity

All eight classifier contrasts against the leading model are reported (Table XV), and the three feature-stack contrasts per season form a second, separately declared family, each on the same 29 district identifiers with one joint resample of those identifiers for both models, 5,000 replicates, seed 20260901.
The multiplicity family is declared as one season with four contrasts, held for both seasons rather than chosen after seeing which framing survives, and Holm's step-down adjustment is applied within it.
Zero differences and tied absolute differences are reported per contrast, and the asymptotic variant is named because ties are present in every contrast.

The winter conclusion is now stated as a negative: Smile GTB is not separated from SVM (Holm p = 0.125) or Random Forest (Holm p = 0.125).
The manuscript states explicitly that this is not equivalence and that no equivalence margin was specified or tested.

## 9. Code-level corrections

| Check | What was wrong | What it says now |
| --- | --- | --- |
| Texture footprint | Described as a 3x3 window | `glcmTexture(size=3)` is a radius, so the footprint is 7x7, averaged over four directional offsets |
| Quantisation | Unstated | B8 mapped over reflectance [0, 0.5] and VV over [-25, 5] dB, clamped, times 31, cast to byte: integer levels 0-31 |
| Scaling bounds | Described but not published | Table V lists all twelve bounds and their source (winter p99 of the labelled sample) |
| Smile GTB settings | Only three arguments named | Sampling rate 0.7 and seed 0 named as Earth Engine defaults, and the three searched arguments described as the capacity controls rather than the whole interface |
| Band lineage | "B2 and B3 survive inside BSI, IBI, BAEI and NDWI" | B3 survives through NDWI and IBI, both selected; B2's only routes are BSI and BAEI, neither of which is selected, so B2 reaches no reported classification |
| Sampling | Unasserted | All 5,000 rows survive the notNull filter in both seasons, 1,000 per class; 4,998 distinct coordinates and 4,988 distinct 10 m cells |
| GHSL epoch | "latest epoch available" | P2023A, 2018 observation epoch |
| OPERA grid | Implied 10 m | 30 m posting, stated as the one consensus condition that cannot resolve a 10 m cell |
| Cloud flag | Correct already | Verified: QA60 bits 10 and 11, mask where either is set |
| Radar ratio | Correct already | Verified: VV(dB) - VH(dB), not called a topographic correction |
| Filtered output | Raw-only accuracy stated | Also stated that the delivered filtered raster has no measured agreement figure and yields different class areas |

## 10. Wording replacements applied

Title, abstract, contribution paragraph, the ESRI paragraph, the seasonal-causation paragraph, the test-use statement, the classifier-ranking wording, the distance-regression wording, the protocol-comparison interpretation, the feature-family paragraph, the feature-reduction paragraph, the limitations and the data-availability statement were replaced along the lines the guide gives, with the numbers taken from the archive rather than from the guide's placeholders.

Removed: "depending only on how they are checked"; the claim that the area-weighted result is directly comparable to random-split OA; the deep-learning accuracy comparison, since no CNN was trained; the unqualified physical readings of the ablation; "second in summer".
The WorldCover method attribution was corrected from Random Forest to a gradient-boosted ensemble, and De Luca et al. (2022) was added as the closest prior design.

The limitations section grew from five items to seven, and now includes the retained-metadata gap, the seasonal-composition caveat and the filtered-output caveat.

## 11. Superseded figures

`figures/fig9.jpg` (the stale CART confusion matrices) and `figures/jabalpur_rebuilt_final.png` (the previous Fig. 1) are still in the repository but are referenced by nothing.
The guide asks for the stale matrix images to be removed; deleting them is left to the authors, since keeping them is the record of what the correction fixed.
`new.tex` now uses `fig_protocols.png` and `fig1_jabalpur_map.png` in their place.

## 12. What was not done, and why

**No independent reference audit.** The guide's fifth item, and the one that would change the answer rather than the wording, is a probability-sampled set of independently interpreted reference locations drawn from both consensus-accepted and consensus-rejected areas.
That requires dated high-resolution interpretation by two readers and has not been carried out.
The manuscript is therefore scoped to conditional map agreement throughout: every reported figure is agreement with a consensus of public maps over a stated domain, and it says so in the abstract, the accuracy section, the limitations and the conclusion.
The illustrative design and budget the guide describes are recorded as the first future direction.

**No release of the training labels.** They are Earth Engine feature collections under a private asset root.
The data-availability section states that the imagery being public is not sufficient for reproduction, and that the study-specific materials are not currently released.

**No multi-seed sensitivity on the feature-stack result.** The guide offers it as affordable-if-possible rather than required; it was not run, and no result is claimed for it.

**No conversion to the 55 current districts, no CNN, and no statewide ground plots**, per the guide's own list of what is not mandatory.
