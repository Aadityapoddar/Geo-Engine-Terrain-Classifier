# Madhya Pradesh Accuracy Study

Evaluation of all five classifiers against ESA WorldCover across the whole of Madhya Pradesh (308,776 km²).
Models are trained on the 4,000 manually labelled Jabalpur points (1,000 per class).

## Method

The classifier is trained on the Jabalpur point set, then applied across MP and compared against ESA WorldCover 10 m as reference.
MP was divided into a 4x3 grid of tiles; each tile was composited and sampled independently, because a single state-wide Earth Engine request exceeds the interactive compute limit.
This produced 2,619 evaluation points plus 720 cropland points.

WorldCover classes were mapped onto the project's four classes only where the mapping is unambiguous.

| WorldCover | project class |
|---|---|
| 10 Tree cover | Forest |
| 80 Permanent water | Water |
| 50 Built-up | Buildings |
| 60 Bare / sparse vegetation | Soil |

Cropland (class 40) is deliberately excluded from the confusion matrices.
In a March-April composite a field may be bare or green, so it has no honest mapping onto four classes, and forcing one would arbitrarily flatter or punish the model.
It is reported separately instead.

Composite window is 2025-03-01 to 2025-04-30, cloud threshold 15%, matching the application defaults.

## Overall ranking

| model | overall | kappa | macro-F1 | Forest F1 | Water F1 | Buildings F1 | Soil F1 |
|---|---|---|---|---|---|---|---|
| SVM (RBF) | **69.3%** | **0.587** | **0.661** | 0.77 | 0.83 | 0.70 | 0.34 |
| KNN (k=5) | 66.3% | 0.548 | 0.628 | 0.75 | 0.82 | 0.66 | 0.27 |
| Gradient Boost | 64.0% | 0.516 | 0.596 | 0.71 | 0.83 | 0.64 | 0.20 |
| Random Forest | 61.4% | 0.482 | 0.570 | 0.73 | 0.84 | 0.61 | 0.10 |
| CART | 61.4% | 0.482 | 0.579 | 0.70 | 0.85 | 0.59 | 0.19 |

Random Forest is the application's default model and is tied for last place.
`MODEL_METADATA` advertises it as 99.35% and "highest internal baseline accuracy", which describes held-out points inside the original training polygon and does not survive contact with the wider state.

## What goes right

**Water is the strongest class by a wide margin.**
Recall is 83-87% and precision 79-86% across every model, and the ranking barely moves between classifiers.
Open water is spectrally unambiguous in Sentinel-2, with near-zero NIR reflectance, so all five models find it regardless of algorithm.

**Forest precision is high.**
Between 85% and 93% of pixels called Forest really are tree cover.
When the model claims forest, it is generally right.

**SVM generalises best.**
It leads on every aggregate metric and degrades most gracefully with distance from the training area.
This is consistent with an RBF kernel producing smoother, better-bounded decision regions than shallow trees.

**The pipeline itself is sound.**
Label values, class ordering, palette, area arithmetic and the tile/export path were all verified correct.
The residual errors are genuine classification errors, not indexing or bookkeeping faults.

## What goes wrong

### 1. Soil is broken, and it is the same failure everywhere

`Soil -> Buildings` is the single largest error in all five models.

| model | Soil recall | Soil -> Buildings | Buildings precision |
|---|---|---|---|
| Random Forest | 8% | 422 | 49% |
| CART | 14% | 396 | 45% |
| Gradient Boost | 15% | 363 | 51% |
| KNN | 21% | 332 | 54% |
| SVM | 27% | 290 | 58% |

Bare ground is being read as built-up.
The consequence is that roughly half of everything labelled "Buildings" across MP is not built-up at all, which makes the Buildings figure unusable as a built-up area estimate outside Jabalpur.

This is the same defect that produced the original Rajasthan desert result of 68% Buildings over open sand.
The Jabalpur points fix it locally, where Soil recall reaches 95%, but it returns as soon as the model leaves the labelled area.

The physical cause is that bright dry bare soil and concrete are nearly identical in Sentinel-2 reflectance.
Measured signatures: Rajasthan desert red reflectance 0.222 against built-up 0.226, SWIR 0.334 against 0.342.
Nothing in the ten input bands separates them reliably.

### 2. Forest is under-detected in the dry season

`Forest -> Soil` is the second largest error in most models (108-171 occurrences).
Forest recall sits at 57-66% despite high precision.

Madhya Pradesh is dominated by dry deciduous forest, which is leaf-off during the March-April composite window.
Bare canopy over dry ground reads spectrally as bare ground.
This is genuine phenology rather than a code fault, and it means the default date range systematically under-reports forest cover across the state.

### 3. Accuracy decays with distance from the labelled area

Per-tile overall accuracy, arranged west to east (SVM):

| tile column | approx longitude | SVM accuracy |
|---|---|---|
| 0 (west) | 74.0-76.2 E | 62% |
| 1 | 76.2-78.4 E | 67% |
| 2 | 78.4-80.6 E | 69% |
| 3 (east) | 80.6-82.8 E | 76% |

The training points span 79.4-80.6 E.
Accuracy rises monotonically as tiles approach that band.
The worst single tile is northwest MP at 42-57% depending on model, covering the Chambal ravines and Malwa plateau, terrain with extensive bare rock and black cotton soil that has no counterpart in the Jabalpur training set.

### 4. Hyperparameters cost about five points, but explain nothing important

Random Forest variants evaluated on four representative MP tiles:

| RF variant | overall | Soil recall | Buildings precision |
|---|---|---|---|
| app default (maxNodes=10, bag 0.3) | 58.3% | 8% | 50% |
| maxNodes=50 | 62.4% | 12% | 51% |
| maxNodes unlimited | 62.9% | 11% | 50% |
| maxNodes unlimited, bag 0.7 | 63.0% | 10% | 50% |

The default `maxNodes=10` is too restrictive and costs roughly 4.7 points of overall accuracy.
Relaxing it is worthwhile and cheap.
It does not, however, repair Soil recall or Buildings precision, which confirms the Soil failure is a data and class-definition problem rather than a tuning problem.

### 5. Cropland has no defensible answer

MP is heavily agricultural, and cropland is excluded from the matrices because it cannot be mapped honestly onto four classes.
What the models do with it varies wildly:

| model | Forest | Water | Buildings | Soil |
|---|---|---|---|---|
| Random Forest | 11% | 4% | 36% | 48% |
| CART | 9% | 2% | 36% | 53% |
| KNN | 9% | 8% | 23% | 61% |
| Gradient Boost | 7% | 8% | 20% | 65% |
| SVM | 7% | 6% | 19% | 68% |

Random Forest assigns 36% of cropland to Buildings while SVM assigns 19%.
Neither can be called correct, because the class scheme has no slot for cropland.
Since cropland is the largest land type in the state, this ambiguity propagates into every area statistic the dashboard reports for an agricultural region.

### 6. Model agreement shows the ceiling is the data

All five models agree with each other on 71% of evaluation points.
When they agree, they are correct only 76% of the time.

Unanimous agreement being wrong roughly a quarter of the time is the clearest evidence in this study that the limit is set by the training data and the class definitions, not by the choice of classifier.
No amount of model selection or tuning will move that ceiling.

## Recommendations

Ordered by measured benefit per unit of effort.

1. **Change the default model from Random Forest to SVM.** One line, worth about 8 points of overall accuracy.
2. **Relax the Random Forest `maxNodes` from 10.** One line, worth about 5 points for anyone who selects RF.
3. **Correct the accuracy badges in `MODEL_METADATA`.** The published 98.7-99.35% figures describe a held-out split inside the training polygon. Spatially split, the honest figure is 79-91%; across MP it is 61-69%.
4. **Add a Cropland class.** This is the largest single source of unresolvable error in an agricultural state, and it is currently forced into Soil or Buildings arbitrarily.
5. **Label bare soil outside the Jabalpur corridor**, particularly bright and rocky terrain in western MP. The east-west accuracy gradient shows exactly where the gaps are.
6. **Reconsider the default date window for forest.** A leaf-on composite would substantially reduce the `Forest -> Soil` leak, at the cost of more cloud.

## Limitations

ESA WorldCover is itself only approximately 75-80% accurate, so these figures measure agreement rather than absolute truth.
Disagreement is not automatically the model's fault, and the reported numbers should be treated as a ceiling on what can be claimed.

The WorldCover "Bare" class is known to be unreliable in humid regions, so part of the Soil disagreement is attributable to the reference data.
The magnitude of the failure, 8-27% recall, is far larger than that explanation covers.

Sampling was stratified by reference class rather than by natural prevalence, so the overall percentages are balanced-class figures and are not weighted by how common each class actually is in MP.

For context, a spatially blocked split within the Jabalpur training area itself gives 79.2-91.5% depending on split direction, against 94.3% for a random split.
The gap between those two is spatial autocorrelation, and the lower figure is the defensible one.

---

# Addendum: Fixing Soil vs Built-up

Feature engineering only.
Training used the 4,000 hand-labelled Jabalpur points throughout; no labels were added at any stage.
Evaluated on 1,274 points across six MP tiles against ESA WorldCover.

## Attribution (Random Forest, each set measured against BASE)

| Feature set | Overall | Forest | Water | Built | Soil | Built prec. | Soil→Built |
|---|---|---|---|---|---|---|---|
| BASE — original 10 bands | 58.6% | 58% | 82% | 78% | 7% | 49% | 213 |
| + bareness indices | 60.4% | 58% | 83% | 81% | 11% | 51% | 203 |
| + GLCM texture | 65.3% | 55% | 83% | 82% | 37% | 58% | 130 |
| + Sentinel-1 SAR | 70.6% | 61% | 82% | 85% | 50% | 59% | 91 |
| **ALL (shipped)** | **73.9%** | 68% | 82% | 88% | **54%** | 63% | **83** |

## What each addition actually did

**Bareness indices (BSI, UI, IBI, SWIR ratio, BAEI)** contributed least, as expected.
They are algebraic functions of bands the model already had, so a nonlinear classifier could largely represent them already.
Soil recall moved 7% to 11%; the Soil→Built error barely changed.

**GLCM texture fixed precision.**
Buildings precision rose 49% to 58% and Soil→Built fell 213 to 130.
Texture measures local structure rather than brightness, and built-up is rough where bare soil is smooth.
That is information no single pixel contains, which is why it worked where the indices did not.

**Sentinel-1 radar fixed recall.**
Soil recall went 7% to 50% on radar alone, the largest single jump in this study.
Buildings produce a double-bounce return that bare soil physically cannot at any brightness, so radar separates them by mechanism rather than appearance.

Texture and radar address opposite halves of the same confusion, which is why using both beats either alone.

## Model guidance — this reverses the earlier recommendation

On the original 10 bands, SVM led Random Forest by 8 points and switching the default was recommended.
With 27 features that is no longer true.

| Model, ALL features | Overall | Soil recall | Built prec. | Water prec. | Soil→Built |
|---|---|---|---|---|---|
| Random Forest (current default) | 73.9% | 54% | 63% | 86% | 83 |
| SVM (RBF) | 70.1% | 37% | 67% | 63% | 70 |

SVM's Water precision fell from 79% to 63%, which shows up in the product as Jabalpur city centre reporting 38.8% water.
Its `gamma=1.0` was tuned for 10 features, not 27.
Leave the default on Random Forest until that is retuned.

## Remaining shortcomings

1. Arid terrain is still broken and got worse. Thar dunes reads 80.9% Buildings under RF, against 52% from the pre-radar SVM configuration. No labelled bright-arid soil exists in the training set.
2. Soil is still the weakest class at 54% recall, against 95% inside Jabalpur.
3. Buildings precision is 63%, so about a third of reported built-up is not built-up.
4. SVM regressed on Water precision and is currently unsafe to select in the UI.
5. Forest recall is 68%; the dry-season leaf-off cause is untouched.
6. Cropland still has no valid mapping onto four classes.
7. A second satellite dependency now exists, with orbit, incidence angle and speckle effects, plus an untested date-widening fallback and slower classification.
8. All figures are agreement with WorldCover, itself only 75-80% accurate.
9. The UI accuracy badges still advertise 98.7-99.35% and match nothing measured here.
