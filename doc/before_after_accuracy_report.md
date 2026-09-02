# Before and after accuracy across all five models

Generated 2026-08-07 from `doc/assets/seasonal_before_after_results.json` and `doc/assets/water_point_change.json`.

Schema `six-class-19-band-v1`, 19 bands, 6 classes, 7 source assets.
Seasons are winter `2025-01-01` to `2025-02-28` exclusive, summer `2025-03-31` to `2025-04-30` exclusive.

Models compared: Random Forest, SVM, Gradient Boosted Trees, CART, KNN.

## What changed in the classes

Agriculture is the new class, label 5.
Soil already holds label 3 and Sand holds label 4, so 5 was the first free slot; any point arriving with Sand's label 4 in the Agriculture source is remapped to 5 on export.

The class was configured in `backend/config.py` and wired through both frontends, but it never reached a model.
`jabalpur_agriculture_points_updated` was imported into Earth Engine from a drawn geometry and carried no attributes at all: every one of its 1,000 features had empty properties.
The training path samples with `sampleRegions(properties=['label'])` and then drops rows where `label` is null, so all 1,000 Agriculture points were discarded in silence.

The repository's own asset audit catches it once run:

```
ValueError: after:agriculture expected label 5, got ['null']
```

The measured effect on the training population:

| Training population | Forest | Water | Buildings | Soil | Sand | Agriculture | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before the fix | 1,000 | 1,000 | 1,000 | 2,000 | 500 | **0** | 5,500 |
| After the fix | 1,000 | 1,000 | 1,000 | 2,000 | 500 | **1,000** | 6,500 |

And through the live API, over an area of interest drawn directly on the Agriculture training points, the served Random Forest returned **0.00% Agriculture** before the fix.

After the fix the class is learnable: XGB reaches 68.00% Agriculture recall on the held-out winter split.

## What changed in the water points

The Before population keeps the original Jabalpur water geometries inside `district_train_table`; the After population uses `jabalpur_water_points`.
Both hold 1000 points, so a count check says nothing about whether they differ.

Measuring the distance from every After point to its nearest Before point settles it:

| Measure | Value |
|---|---:|
| Points compared | 998 |
| Mean displacement | 3.62 m |
| Median displacement | 3.7 m |
| Maximum displacement | 6.68 m |
| Moved more than 100 m | 0 |
| Half-diagonal of a 10 m pixel | 7.07 m |

Not one point moved further than half the diagonal of a single Sentinel-2 pixel.
That bound is the signature of a table round trip rather than an edit: `sampleRegions(geometries=True)` returns the centre of the pixel it read, not the point it was handed, so exporting a point set and re-importing it displaces every point by up to 7.07 m.

**The water points were not repositioned.**
Before and After hold the same water locations, and the water class contributes no real difference to any before/after comparison in this report.

Whether the points sit on water is a separate question, and both sets answer it identically because they are the same points:

| Season | Sampled | NDWI positive | On JRC surface water |
|---|---:|---:|---:|
| Summer | 455 | 51.87% | 57.14% |
| Winter | 455 | 66.15% | 57.14% |

Two things worth flagging.
Fewer than half the water points yield a valid reading in either season, because the cloud-masked median composite has gaps over open water.
And only around three in five sit on water that JRC has ever observed, which caps how good the Water class can get regardless of the model.

## How to read these numbers

Two scores are reported per run and they measure different things.

**External five-class** compares predictions against a high-confidence consensus of public maps across every Madhya Pradesh district. Soil and Sand collapse into a single Bare class for this comparison because the public references do not separate them. This is agreement with other maps, not field survey.

**Held-out six-class** scores the model against the project's own labels on a fixed spatial-block split, keeping all six classes. Blocks rather than random points, because labelled points come in clusters and a random split would score each model against near duplicates of its own training rows.

Raw held-out overall accuracy falls from Before to After for every model. That is expected and is not a regression. Before was scored on a four-class problem, because Sand and Agriculture had no held-out rows and Agriculture had no training rows either. After is scored on a genuinely harder six-class problem. The like-for-like table restricts scoring to the classes both conditions actually attempt, which is the comparison that answers whether the model got worse at what it already did.

## Measured results

The whole-Madhya-Pradesh external comparison is still computing and is not reported here. The held-out results below are complete for all five models in both seasons.


This report measures agreement with high-confidence public-map consensus; it is not field-survey ground truth. External five-class results use Forest, Water, Buildings, Bare, and Agriculture. Soil and Sand are collapsed to Bare only for that external comparison. Held-out six-class results retain all six project labels.

> The external five-class half of the evidence is still computing and is omitted here rather than partially reported.

## Held-out six-class

### Winter

| Model | Before OA | After OA | Delta (pp) |
|---|---:|---:|---:|
| RF | 88.68% | 65.98% | -22.70 |
| SVM | 75.92% | 63.26% | -12.66 |
| XGB | 89.76% | 75.83% | -13.93 |
| CART | 83.20% | 62.98% | -20.22 |
| KNN | 76.86% | 63.20% | -13.66 |

### Summer

| Model | Before OA | After OA | Delta (pp) |
|---|---:|---:|---:|
| RF | 84.35% | 65.11% | -19.25 |
| SVM | 74.91% | 60.37% | -14.54 |
| XGB | 86.23% | 73.00% | -13.23 |
| CART | 74.77% | 67.34% | -7.43 |
| KNN | 71.38% | 56.83% | -14.55 |

## Held-out accuracy on the shared five classes

Overall accuracy above is not a like-for-like comparison: After is scored on six classes where Before was scored on the four it could actually predict, so it is charged for a question Before was never asked. These figures drop the Agriculture reference rows and keep every prediction column, so an After model that answers Agriculture on a Forest row is still counted wrong.

### Winter

| Model | Before OA | After OA | Delta (pp) | Rows scored |
|---|---:|---:|---:|---:|
| RF | 88.68% | 87.31% | -1.37 | 1387 / 1387 |
| SVM | 75.92% | 67.41% | -8.51 | 1387 / 1387 |
| XGB | 89.76% | 78.37% | -11.39 | 1387 / 1387 |
| CART | 83.20% | 82.05% | -1.15 | 1387 / 1387 |
| KNN | 76.86% | 68.71% | -8.15 | 1387 / 1387 |

### Summer

| Model | Before OA | After OA | Delta (pp) | Rows scored |
|---|---:|---:|---:|---:|
| RF | 84.35% | 82.84% | -1.51 | 1387 / 1387 |
| SVM | 74.91% | 69.65% | -5.26 | 1387 / 1387 |
| XGB | 86.23% | 78.88% | -7.35 | 1387 / 1387 |
| CART | 74.77% | 71.30% | -3.46 | 1387 / 1387 |
| KNN | 71.38% | 69.21% | -2.16 | 1387 / 1387 |

## Agriculture recall

Before has no Agriculture rows at all, which is the point: the class was configured but its source asset carried no label, so it never reached training.

### Winter

| Model | Before recall | After recall | After rows |
|---|---:|---:|---:|
| RF | N/A | 0.22% | 450 |
| SVM | N/A | 50.44% | 450 |
| XGB | N/A | 68.00% | 450 |
| CART | N/A | 4.22% | 450 |
| KNN | N/A | 46.22% | 450 |

### Summer

| Model | Before recall | After recall | After rows |
|---|---:|---:|---:|
| RF | N/A | 10.44% | 450 |
| SVM | N/A | 31.78% | 450 |
| XGB | N/A | 54.89% | 450 |
| CART | N/A | 55.11% | 450 |
| KNN | N/A | 18.67% | 450 |

## Reference limitations

WorldCover, WorldCereal (2021), and GHSL (2018) are temporally mismatched with the 2025 Sentinel composites. Low-confidence or conflicting pixels are masked. Sand has no independent external score.

## Further limitations

Sand has no held-out rows in either condition, so it carries no held-out score at all.
Its training points all fall on the training side of the spatial block split, which means the split does not currently test it.

The five model configurations come from `make_classifier`, so notebook, evaluation and served backend cannot drift apart. They were not re-tuned for six classes; these are the same hyperparameters chosen for the four-class problem, which is the most likely reason Random Forest and CART barely predict Agriculture at all while Gradient Boosted Trees handles it.
