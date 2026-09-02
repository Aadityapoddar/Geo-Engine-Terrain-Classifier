# Feature Band Reference

Every one of the 27 input features used by the classifier, why it is there, and what was left out.

Source of truth is `BANDS` in `backend/config.py:36-39`, constructed by `_add_spectral_indices()` and `_add_sar()` in `backend/gee_classifier.py:57-132`.
Classes are Forest, Water, Buildings and Soil.
Every design decision below is ultimately about one problem: bright dry bare soil and concrete are nearly the same colour to a satellite, and the model kept calling soil "buildings".

## How to read this document

Each band gets four things.

**Formula** is what the code actually computes, not the textbook form, where the two differ.
**Plain English** is what the number means physically, with no remote sensing vocabulary.
**Why it is in** is the specific job it does for these four classes.
**Cost** is what including it charges us, since nothing is free.

## Contents

1. The problem the band stack exists to solve
2. Group 1: Raw Sentinel-2 reflectance, 6 bands
3. Group 2: Classic normalized indices, 4 bands
4. Group 3: Bareness and built-up discriminators, 5 bands
5. Group 4: GLCM texture on optical NIR, 6 bands
6. Group 5: Sentinel-1 radar backscatter, 3 bands
7. Group 6: GLCM texture on radar, 3 bands
8. Measured contribution of each group
9. What is deliberately not included, and why
10. Cross-cutting tradeoffs
11. Known discrepancies with the literature
12. Ranked candidate changes

---

## 1. The problem the band stack exists to solve

Four classes, and three of them are easy.

Water absorbs almost all near-infrared light, so it is close to black in B8 and unmistakable.
It scores 82-85% F1 in every model tested and barely responds to which classifier is used.
Forest is bright in NIR and dark in red because chlorophyll absorbs red light, which is the single most reliable contrast in optical remote sensing.

Soil and Buildings are the entire difficulty.
Measured signatures from the Madhya Pradesh study: Rajasthan desert sand has red reflectance 0.222 against built-up 0.226, and SWIR 0.334 against 0.342.
Those are the same numbers within noise.
A classifier looking only at per-pixel colour has nothing to work with, and the original ten-band model duly assigned 68% of the Thar desert to "Buildings".

So the band stack is built in layers, each attacking that confusion from a different physical angle:

- **Colour** (Groups 1 and 2) establishes the three easy classes and provides the raw material for everything else.
- **Spectral shape** (Group 3) asks whether the SWIR curve bends like a mineral or like concrete, rather than asking how bright the pixel is.
- **Structure** (Groups 4 and 6) asks whether the neighbourhood is rough or smooth, which is information no single pixel contains.
- **Physical mechanism** (Group 5) asks whether the surface bounces radar twice off a vertical wall, which soil cannot do at any brightness.

Each layer is a different question, which is why they compose rather than duplicate.

---

## 2. Group 1: Raw Sentinel-2 reflectance, 6 bands

Source: `COPERNICUS/S2_SR_HARMONIZED`, cloud-masked via the QA60 bitmask and divided by 10000 to surface reflectance, median-composited over the requested window (`gee_classifier.py:49-54`, `:135-165`).
These six are the only raw measurements in the stack.
The other 21 features are derived from them, from radar, or from their spatial arrangement.

### B2 - Blue, 492 nm, 10 m

**Formula:** raw surface reflectance, no transformation.

**Plain English:** how much blue light the ground sends back.

**Why it is in:** blue is the shortest usable wavelength and therefore the most scattered by haze, which makes it the band where atmosphere and turbidity show up first.
It is what lets turbid water separate from clean water, and it is a term in BSI, where it acts as the counterweight that stops bright bare ground from being confused with bright vegetation.

**Cost:** it is the noisiest of the four 10 m bands because of atmospheric scattering, and residual haze in a median composite lands here first.

### B3 - Green, 560 nm, 10 m

**Formula:** raw surface reflectance.

**Plain English:** how much green light comes back. Healthy plants reflect a little more green than red or blue, which is why they look green.

**Why it is in:** it is the numerator of NDWI, so it carries the water class, and it appears in both IBI and BAEI as the vegetation-and-water suppression term.
On its own it is a weak discriminator, but it is load-bearing inside three derived indices.

**Cost:** low. It is nearly redundant with B2 and B4 for bare surfaces, contributing mostly through the indices rather than directly.

### B4 - Red, 665 nm, 10 m

**Formula:** raw surface reflectance.

**Plain English:** how much red light comes back. Chlorophyll eats red light, so living plants are dark here and everything else is not.

**Why it is in:** it is half of NDVI and half of SAVI, which together carry the entire Forest class.
It also appears in BSI and BAEI as the bare-surface brightness term.
This is the busiest single band in the stack, feeding four derived features.

**Cost:** none worth mentioning. It would be indefensible to omit it.

### B8 - Near-infrared, 833 nm, 10 m

**Formula:** raw surface reflectance.

**Plain English:** light just past what the eye can see. Leaf cell structure reflects it strongly, so vegetation is very bright here, and water is almost perfectly black.

**Why it is in:** the most informative band in the stack.
It is the other half of NDVI, NDWI, NDBI, SAVI, BSI, UI and IBI, and it is the band the optical GLCM texture is computed on.
It carries Water by absorption and Forest by reflection at the same time.

**Cost:** none. Removing B8 would collapse the model.

### B11 - Short-wave infrared 1, 1610 nm, 20 m

**Formula:** raw surface reflectance, resampled to 10 m by Earth Engine at sample and classify time.

**Plain English:** the wavelength where water inside a surface shows itself. Wet things are dark here, dry things are bright, regardless of what colour they are to the eye.

**Why it is in:** the moisture axis, and the single most useful non-visible band for the soil-versus-built problem.
It drives NDBI, BSI, IBI, SWIRratio and BAEI.
Bare dry soil and concrete are both bright in B11, but they get there differently, and the ratio bands in Group 3 exist to read that difference.

**Cost:** native resolution is 20 m, so it is upsampled.
Every feature built from B11 therefore carries a real 20 m footprint no matter what the pixel grid says, which blurs small built-up patches and is one reason isolated rural buildings are missed.

### B12 - Short-wave infrared 2, 2190 nm, 20 m

**Formula:** raw surface reflectance, resampled to 10 m.

**Plain English:** further into the infrared, where specific minerals leave fingerprints. Clay and carbonate absorb strongly here, so soil dims while concrete and asphalt stay bright.

**Why it is in:** this is the mineralogy band and the reason `SWIRratio` works at all.
It is also the driver of UI.
Of all six raw bands this is the one that most directly attacks the target confusion, because the divergence between soil and concrete is larger at 2.2 µm than anywhere else in the Sentinel-2 range.

**Cost:** 20 m native, same upsampling caveat as B11, and it is the noisiest SWIR band because atmospheric absorption is stronger there.

---

## 3. Group 2: Classic normalized indices, 4 bands

Computed in `_add_spectral_indices()` at `gee_classifier.py:65-72`.
All four are normalized differences, which means they range from -1 to +1 and are invariant to overall illumination.
That invariance is the point: a hillside in shadow and the same surface in sun give the same index value even though their raw reflectances differ.

### NDVI - Normalized Difference Vegetation Index

**Formula:** `(B8 - B4) / (B8 + B4)`

**Plain English:** how much more infrared than red a surface sends back. Living plants send back far more infrared, so high means green and growing, near zero means bare, negative means water.

**Why it is in:** the single best-established index in remote sensing (Rouse et al., 1973) and the one the reference paper measured as its best-performing index addition with SVM.
It carries the Forest class almost single-handed and pushes water strongly negative, so it separates two of four classes by itself.

**Cost:** it saturates over dense canopy, so it cannot distinguish moderately dense forest from very dense forest.
That does not matter here because Forest is a single class.
It is also unhelpful in the dry season, when leaf-off deciduous forest reads as bare ground, which is the documented cause of the 68% Forest recall in the MP study.

### NDWI - Normalized Difference Water Index

**Formula:** `(B3 - B8) / (B3 + B8)`

**Plain English:** green light minus infrared. Water reflects a bit of green and essentially no infrared, so open water goes strongly positive and everything else goes negative.

**Why it is in:** it is McFeeters' (1996) formulation, and it makes the Water class trivially separable.
Water is the strongest class in the model at 82-85% F1 and this index is the reason.

**Cost:** the McFeeters form is known to leak on built-up surfaces, which can also show positive values.
The MNDWI variant using B11 instead of B8 handles that better, and is discussed in the exclusions section.
Note that the reference paper found NDWI added nothing on top of NDVI for its classes, though its classes did not include a general water class the way ours does.

### NDBI - Normalized Difference Built-up Index

**Formula:** `(B11 - B8) / (B11 + B8)`

**Plain English:** short-wave infrared minus near infrared. Built surfaces reflect more SWIR than NIR; vegetation does the opposite.

**Why it is in:** the standard built-up index (Zha et al., 2003) and the first thing anyone reaches for on this problem.
It is genuinely good at separating built-up from vegetation.

**Cost:** honesty requires saying that NDBI is weak at exactly the job this project needs.
It separates built-up from vegetation, not built-up from bare soil, because dry bare soil is also SWIR-bright and also lands positive.
NDBI was in the original ten-band model that assigned 68% of the Thar desert to Buildings.
It stays because it is a strong vegetation-versus-nonvegetation feature, not because it solves the target problem.

### SAVI - Soil Adjusted Vegetation Index

**Formula:** `1.5 * (B8 - B4) / (B8 + B4 + 0.5)`, that is L = 0.5 and the scaling factor (1 + L)

**Plain English:** NDVI with a correction for the fact that bright ground showing through thin vegetation makes plants look less green than they are.

**Why it is in:** Huete's (1988) soil-line correction matters here specifically because a large fraction of the study area is sparse dry-season vegetation over bright soil, which is the exact regime where NDVI misreads.
Where NDVI and SAVI disagree, the size of the disagreement is itself a soil-brightness signal the classifier can use.

**Cost:** it is strongly correlated with NDVI, so it adds a dimension without adding much independent information.
The fixed L = 0.5 is a middle-of-the-road choice; the correct L varies with actual canopy density, and no tuning was done.
This is one of the more defensible candidates for removal if the feature count ever needs to come down.

---

## 4. Group 3: Bareness and built-up discriminators, 5 bands

Computed at `gee_classifier.py:74-85`.
These five were added specifically for the soil-versus-buildings failure.
They are all algebraic functions of the same six raw bands, so strictly they add no new measurement.
What they add is a change of coordinates that puts the decision boundary somewhere an axis-aligned tree split or a distance-based kernel can actually find it.

The measured result was the weakest of the three additions: overall accuracy 58.6% to 60.4%, Soil recall 7% to 11%.
That is a real but small gain, and the reason it is small is precisely that a nonlinear classifier could already represent most of these combinations internally.
They are kept because the gain is real, the compute cost is nil, and they make the model's reasoning legible.

### BSI - Bare Soil Index

**Formula:** `((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))`

**Plain English:** adds up the two wavelengths where bare ground is bright and subtracts the two where vegetation and water are bright. High means nothing is growing here.

**Why it is in:** it cleanly divides bare surfaces from vegetated and wet ones.
It does not separate soil from concrete, since both are strongly positive.
Its job is to collapse the four-way decision into a two-way one, so that the harder features downstream only have to arbitrate soil versus built.

**Cost:** the reference paper found BSI reduced accuracy every time it was included, for both MLC and SVM.
That result does not transfer, because the paper's imagery was four-band Planet with no SWIR at all, so its BSI was a substitute formulation and not the same quantity as the one above.
Their finding is not evidence against this band.

### UI - Urban Index

**Formula:** `(B12 - B8) / (B12 + B8)`

**Plain English:** the same idea as NDBI, but reading the far infrared instead of the near one.

**Why it is in:** it looks redundant with NDBI and is not.
Concrete and asphalt hold their reflectance out to 2.2 µm, while soil falls off there because of clay hydroxyl and carbonate absorption.
NDBI reads 1.6 µm where the two materials are similar; UI reads 2.2 µm where they diverge most.
Same algebra, different physics, genuinely different information.

**Cost:** highly correlated with NDBI, so it is a near-duplicate axis for any classifier that only looks at ranking rather than exact value.
It also inherits B12's noise and its 20 m native resolution.

### IBI - Index-based Built-up Index

**Formula:** built in three steps at `gee_classifier.py:80-82`

```
t1  = 2 * B11 / (B11 + B8)                        an NDBI-equivalent
t2  = B8 / (B8 + B4)  +  B3 / (B3 + B11)          a SAVI-like term plus an MNDWI-like term
IBI = (t1 - t2) / (t1 + t2)
```

**Plain English:** take the built-up signal, then explicitly subtract the vegetation signal and the water signal. What survives is built-up and only built-up.

**Why it is in:** Xu's (2008) index is the most surgical built-up feature in the stack.
The others say "this is bare" or "this is bright in SWIR"; IBI says "this is bare and it is not vegetation and it is not water".
Note that it smuggles in an MNDWI term via `B3/(B3+B11)`, which is the only place in the entire stack the superior MNDWI water formulation appears.

**Cost:** it is a ratio of ratios, so error in any of four bands propagates twice and it is the numerically touchiest feature in the group.
Its output range is compressed near zero for most land, which makes it a weak split candidate for shallow trees, and `maxNodes=10` on the default Random Forest is shallow.

### SWIRratio

**Formula:** `B11 / (B12 + 1e-6)`

**Plain English:** compares the two infrared bands against each other rather than against a brightness reference. Because it is a plain ratio, how bright or shadowed the pixel is cancels out entirely, leaving only the shape of the reflectance curve.

**Why it is in:** this is the answer to the measured tie at the heart of the whole project.
Dry soil red reflectance 0.222 against concrete 0.226 is a brightness tie that no brightness-based feature can break.
SWIRratio does not look at brightness.
Clay-rich soil has a pronounced absorption dip at 2.2 µm that pushes B11/B12 well above 1; concrete and asphalt have a much flatter SWIR curve and sit closer to 1.
Of the five bands in this group, this is the most direct attack on the target confusion.

**Cost:** an unbounded ratio rather than a bounded normalized difference, so its numeric range is wide and asymmetric, which matters for the distance-based models.
The `1e-6` guard prevents division by zero but does nothing for near-zero denominators, where the value can spike.
Both SWIR bands are 20 m native, so this is effectively a 20 m feature.

### BAEI - Built-up Area Extraction Index

**Formula:** `(B4 + 0.3) / (B3 + B11 + 1e-6)`

**Plain English:** red light against the sum of green and short-wave infrared, with a fixed offset. Tuned empirically so built-up lands in a distinct band of values from bare ground.

**Why it is in:** Bouzekri et al. (2015) published this index specifically to separate built-up from bare land, which is not what most built-up indices are optimised for.
Most, including NDBI, are optimised to separate built-up from vegetation.
That makes BAEI the most on-target published index available for this exact failure mode.

**Cost:** the 0.3 is an empirical constant fitted on Algerian arid terrain, and it is being applied unchanged to central India.
There is no reason to believe 0.3 is the right offset here and it has never been tuned.
Like SWIRratio it is an unbounded ratio with the same numeric-range concerns.

---

## 5. Group 4: GLCM texture on optical NIR, 6 bands

Computed at `gee_classifier.py:87-93`.
The B8 band is rescaled from 0-0.5 reflectance into 0-31 integers, giving 32 grey levels, then `glcmTexture(size=3)` computes a grey level co-occurrence matrix averaged over four directions.

A GLCM counts how often each pair of grey values occurs side by side.
Smooth surfaces produce pairs that are nearly equal, so the counts concentrate on the diagonal.
Rough surfaces produce mismatched pairs and the counts spread out.
Every metric below is a different way of measuring how spread out that matrix is.

This is the first group that measures something no single pixel contains.
Measured contribution: overall 60.4% to 65.3%, Buildings precision 49% to 58%, and Soil-called-Buildings errors 213 down to 130.
Texture fixed precision, which is exactly what it should fix: it stops the model calling smooth things buildings.

In the notation below, `p(i,j)` is the probability of grey level `i` sitting next to grey level `j`, and `µ` and `s` are the mean and standard deviation of the matrix margins.

### g_contrast - Contrast

**Formula:** `S (i - j)^2 * p(i,j)`

**Plain English:** how big the jumps in brightness are between neighbouring pixels, with big jumps counted much more heavily than small ones because of the square.

**Why it is in:** roof edges, walls and building shadows create large local brightness steps.
Bare soil, however bright, is locally smooth and has small steps.
This is the most direct expression of "built-up is rough and soil is not" in the whole stack.

**Cost:** it is also large along field boundaries, riverbanks and forest edges, so it produces false built-up signal at any sharp linear boundary.

### g_ent - Entropy

**Formula:** `-S p(i,j) * log p(i,j)`

**Plain English:** how disorderly the neighbourhood is. A patch where you cannot predict the next pixel from the current one scores high; a uniform patch scores near zero.

**Why it is in:** urban fabric is disorderly at 10 m: roofs, roads, gaps, shadows and vegetation interleave within a few pixels.
Bare soil and open water are highly predictable.
Entropy captures that unpredictability without caring about the direction or size of the changes, so it is complementary to contrast rather than duplicative.

**Cost:** it is sensitive to the 32-level quantization choice; a different grey-level count shifts entropy values systematically.
It also rises with sensor noise, so dark low-signal areas can read as spuriously textured.

### g_var - Variance

**Formula:** `S (i - µ)^2 * p(i,j)`

**Plain English:** how much the brightness varies across the whole neighbourhood, regardless of whether neighbouring pixels differ.

**Why it is in:** it captures broader heterogeneity that pixel-pair metrics miss, such as a mixed patch of building and garden that is internally smooth but overall varied.

**Cost:** this is the weakest member of the group on published evidence.
The reference paper ranked variance the worst of all eight texture features with MLC, and only third with SVM.
It is also the most correlated with plain brightness of the six, so it partially duplicates B8 itself.
This is the first band to consider dropping if a slot is needed, and the section on discrepancies proposes swapping it for correlation.

### g_idm - Inverse Difference Moment, also called homogeneity

**Formula:** `S p(i,j) / (1 + (i - j)^2)`

**Plain English:** the opposite of contrast. It scores high when neighbouring pixels are similar, so smooth surfaces score high and rough ones score low.

**Why it is in:** it is not simply the inverse of contrast, because the weighting is different.
Contrast squares the difference and so is dominated by rare extreme jumps; IDM weights by the reciprocal and so is dominated by the common small differences.
One is an outlier detector, the other describes the bulk.
For this problem IDM is the positive evidence for soil, where contrast is the positive evidence for buildings, and having both means the classifier can distinguish "definitely smooth" from "merely not rough".

**Cost:** near-saturated over any uniform surface, so it cannot tell smooth soil from smooth water from smooth bare rock.

### g_diss - Dissimilarity

**Formula:** `S |i - j| * p(i,j)`

**Plain English:** the same measurement as contrast but using the plain difference instead of the squared difference, so one dramatic edge does not dominate.

**Why it is in:** it is contrast without the outlier sensitivity.
Dense low-rise settlement produces many moderate edges and few extreme ones, which reads as ordinary under contrast but clearly textured under dissimilarity.
Since much of the built-up in Madhya Pradesh is exactly that sort of low-rise fabric, this band handles a case the squared metric underweights.

**Cost:** genuinely and heavily correlated with contrast.
Of the six, this is the most defensible to call redundant, and the argument for keeping it rests on the low-rise case above rather than on measured attribution.

### g_asm - Angular Second Moment, also called energy

**Formula:** `S p(i,j)^2`

**Plain English:** how concentrated the pattern is on a few repeating pixel pairings. High means the patch is uniform or strictly repetitive; low means anything goes.

**Why it is in:** it is entropy's mathematical counterpart and behaves differently in the middle of the range even though the two agree at the extremes.
ASM responds to regular repeating structure, and regular repeating structure at 10 m is a strong built-up signature: planned housing, industrial roofing and street grids all repeat.
Natural bare ground is irregular without being repetitive.

**Cost:** substantially redundant with entropy, and the marginal information sits in a narrow middle band of values.
Second candidate for removal after `g_var`.

---

## 6. Group 5: Sentinel-1 radar backscatter, 3 bands

Computed in `_add_sar()` at `gee_classifier.py:96-132` from `COPERNICUS/S1_GRD`, IW mode, dual polarisation, median-composited.

This is the most important group in the stack and it was the largest single measured gain in the entire project: Soil recall went from 7% to 50% on radar alone, and overall accuracy 65.3% to 70.6%.

Radar is not a camera.
It sends its own microwave pulse and measures what comes back, so it does not care about sunlight, colour or cloud.
What it measures is geometry and moisture.
A flat surface reflects the pulse away from the satellite and appears dark; a rough surface scatters some back; a vertical wall meeting flat ground forms a corner that bounces the pulse twice and returns it almost perfectly to the sensor, which is called double bounce and which appears very bright.

Bare soil has no vertical walls.
Buildings are made of them.
That is a difference in physical mechanism rather than in appearance, and it is the reason this group succeeded where colour-based features failed.

All three bands are rescaled from decibels into roughly 0-1, because the RBF kernel in the SVM is distance-based and an unscaled dB range of -25 to +5 would otherwise dominate every reflectance feature in the kernel.

### VV - Vertical transmit, vertical receive

**Formula:** `unitScale(VV_dB, -25, 5)` clamped to 0-1

**Plain English:** send a vertically oriented pulse, listen for the vertically oriented return. This is the channel that lights up on vertical structures.

**Why it is in:** VV is where double bounce shows itself most strongly, so it is the primary built-up detector in the stack.
It is also the band the radar texture in Group 6 is computed from.
Smooth dry soil returns very little in VV and open water returns almost nothing, so this band separates buildings from both at once.

**Cost:** it is sensitive to surface roughness and moisture in ways that have nothing to do with buildings.
Ploughed fields, wet soil after rain and rocky ground all raise VV.
The scaling bounds of -25 and 5 dB are hard-coded and were not calibrated against the study area's actual distribution.

### VH - Vertical transmit, horizontal receive

**Formula:** `unitScale(VH_dB, -30, 0)` clamped to 0-1

**Plain English:** send a vertical pulse, listen for a horizontal return. Getting a signal back in the wrong orientation means it bounced around inside something complicated before escaping.

**Why it is in:** cross-polarised return comes from volume scattering, which is what happens inside a tree canopy where the pulse ricochets among branches and leaves.
VH is therefore the forest detector, and it works in the dry season when leaf-off canopy defeats NDVI.
It is also low over smooth soil, so it contributes to the primary separation as well.

**Cost:** noisier than VV, because cross-polarised return is intrinsically weaker and closer to the noise floor.
Over very smooth surfaces such as calm water it can hit the sensor noise floor entirely, which makes it unreliable rather than merely low.

### VVVH - Co- to cross-polarisation ratio

**Formula:** `unitScale(VV_dB - VH_dB, -5, 20)` clamped to 0-1. Subtracting in decibels is dividing in linear power.

**Plain English:** how much of the return kept its original orientation. High means the signal bounced off hard flat and vertical surfaces; low means it got scrambled inside vegetation.

**Why it is in:** the ratio cancels out everything that scales both channels equally, including terrain slope effects, incidence angle variation and overall surface brightness.
What survives is the scattering mechanism itself.
This is the radar analogue of what SWIRratio does in the optical domain: strip away brightness, keep the physics.
It is the cleanest built-up-versus-vegetation feature in the stack.

**Cost:** the scaling window of -5 to 20 dB is wider than the typical real range of roughly 5-12 dB, so most real values are compressed into the middle of the 0-1 output and the feature is less discriminative than it could be.
This is a calibration knob that has not been turned.

---

## 7. Group 6: GLCM texture on radar, 3 bands

Computed at `gee_classifier.py:130-131`, the same GLCM machinery from Group 4 applied to the VV band instead of B8.

The reasoning is that structure and mechanism are separate signals and stacking them beats either alone.
An urban area is bright in VV because of double bounce, and it is also spatially chaotic in VV because buildings, roads and gaps alternate every few pixels.
A rocky hillside can be equally bright in VV, but its brightness is spatially smoother.
Radar texture separates those two cases where radar brightness alone cannot.

Only three of the six optical metrics were carried over, on the grounds that radar is noisier and speckle-prone, so the fine-grained distinctions between correlated metrics are less trustworthy than they are in the optical domain.

### s_contrast - Radar contrast

**Formula:** as `g_contrast`, computed on VV quantized to 32 levels

**Plain English:** how sharply radar brightness changes between neighbouring pixels.

**Why it is in:** built-up produces alternating very bright double-bounce returns and very dark radar shadows within a few pixels, which is the largest local contrast of any land cover type.
This is the strongest of the three radar texture bands.

**Cost:** speckle, the grainy interference noise inherent to coherent radar imaging, also produces high contrast.
The median composite over time suppresses speckle but does not remove it, so some of this band's signal in low-scene-count areas is noise.

### s_var - Radar variance

**Formula:** as `g_var`, computed on VV

**Plain English:** how much radar brightness varies across the neighbourhood overall.

**Why it is in:** it captures settlement-scale heterogeneity, a mixed patch of buildings and open ground that is not sharply edged but is far from uniform.

**Cost:** the same objection as `g_var`, made worse by speckle.
This is the weakest band in the entire 27 and the most defensible single removal.

### s_ent - Radar entropy

**Formula:** as `g_ent`, computed on VV

**Plain English:** how unpredictable the radar pattern is from one pixel to the next.

**Why it is in:** natural surfaces scatter radar in ways that are statistically consistent across a patch, because the physical process generating the return is the same everywhere.
Built environments mix several different scattering mechanisms within metres, which shows up as high entropy.
This is a mechanism-diversity measure and it has no optical equivalent.

**Cost:** speckle raises entropy directly, so scenes built from few Sentinel-1 acquisitions will have inflated values here.

---

## 8. Measured contribution of each group

From the addendum to the Madhya Pradesh study, 1,274 evaluation points across six MP tiles, Random Forest, trained throughout on the same 4,000 hand-labelled Jabalpur points with no labels added.

| Feature set | Overall | Soil recall | Built precision | Soil called Built |
|---|---|---|---|---|
| Base, original 10 bands | 58.6% | 7% | 49% | 213 |
| plus bareness indices | 60.4% | 11% | 51% | 203 |
| plus GLCM texture | 65.3% | 37% | 58% | 130 |
| plus Sentinel-1 SAR | 70.6% | 50% | 59% | 91 |
| **All 27, shipped** | **73.9%** | **54%** | **63%** | **83** |

Read that as a hierarchy of usefulness.

The bareness indices bought 1.8 points, which is the least, and that is expected: they are algebraic rearrangements of bands the model already had, and a nonlinear classifier can represent most of them internally.
GLCM texture bought 4.9 points and, more importantly, moved Buildings precision, because it stops smooth things being called buildings.
Radar bought 5.3 points and moved Soil recall from 7% to 50%, because it identifies buildings by a mechanism soil cannot imitate.

Texture and radar attack opposite halves of the same confusion, precision and recall respectively, which is why using both beats either.

One consequence worth recording: with 27 features Random Forest now beats SVM (73.9% against 70.1%), reversing the ranking that held on 10 bands.
SVM's `gamma=1.0` was tuned for a 10-dimensional feature space and has not been retuned for 27.
The `MODEL_METADATA` accuracy figures in `backend/config.py:53-111` predate all of this and match nothing measured here.

---

## 9. What is deliberately not included, and why

### 9.1 Sentinel-2 bands not used

Sentinel-2 carries 13 bands. Six are used.

**B1, coastal aerosol, 443 nm, 60 m.** Designed for atmospheric correction and coastal water, not land cover.
At 60 m it is six times coarser than the classification grid, so including it would smear a 60 m footprint across the decision.
The Level-2A product has already used it internally for atmospheric correction, so its information is present in the corrected bands already.

**B5, B6, B7, red edge, 705 / 740 / 783 nm, 20 m.** These are the strongest bands in the whole instrument for vegetation, capturing the sharp rise from red absorption to NIR reflection and shifting measurably with chlorophyll content and plant stress.
They are excluded because our Forest class is a single undifferentiated lump.
Red edge distinguishes species, stress and crop type, none of which the four-class scheme asks for, and Forest is already the second-strongest class.
Three extra 20 m bands to improve a class that is not the bottleneck is a bad trade.
This would change immediately if a Cropland class were added, which is recommendation 4 of the MP study: red edge is the standard tool for separating crops from natural vegetation, and cropland is currently the largest unresolvable error in the model.

**B8A, narrow NIR, 865 nm, 20 m.** Nearly the same measurement as B8 at half the resolution.
Its narrower bandwidth avoids a water vapour absorption feature that B8 clips, which matters for precise biophysical retrieval and does not matter for four-class discrimination.
Straightforwardly redundant.

**B9, water vapour, 945 nm, 60 m.** Measures atmosphere, not ground. 60 m. Nothing to contribute.

**B10, cirrus, 1375 nm, 60 m.** Not a choice: B10 does not exist in the Level-2A surface reflectance product, having been consumed during atmospheric correction.

**QA60.** Used, but as a mask rather than a feature, at `gee_classifier.py:49-54`.
Feeding cloud-flag bits to a classifier as a predictor invites it to learn "cloudy pixels are class X".

### 9.2 Spectral indices not used

**MNDWI, `(B3 - B11) / (B3 + B11)`.** This is the better water index.
Xu (2006) introduced it precisely because McFeeters' NDWI leaks on built-up surfaces, and built-up leakage is this project's central problem.
It is not included as a standalone band, which is arguably an oversight.
The partial defence is that its algebraic form already appears inside IBI as the `B3/(B3+B11)` term, so the classifier has indirect access, and Water is not a failing class at 82-85% F1.
Still the strongest omission on this list.

**EVI.** A more robust vegetation index than NDVI in high-biomass conditions, using blue to correct for aerosols.
Excluded because it is a third strongly correlated vegetation index after NDVI and SAVI, and vegetation is not the bottleneck.
Adding it would worsen the collinearity problem for no measured gain.

**NDMI or NDWI-Gao, `(B8 - B11) / (B8 + B11)`.** Vegetation water content.
This is arithmetically just NDBI with the sign flipped, so it is perfectly anti-correlated with a band already present and carries literally zero additional information for any classifier invariant to sign.

**NBR, normalized burn ratio, `(B8 - B12) / (B8 + B12)`.** Same objection: it is UI negated.

**NDSI, snow index.** No snow in the study area.

**NDBaI, NBAI, BRBA and the rest of the bareness index literature.** There are dozens.
They are all built from the same six bands and are mutually correlated to a degree that makes the choice among them nearly arbitrary.
Five were chosen for coverage of distinct formulations, normalized difference (BSI, UI), index-of-indices (IBI), pure ratio (SWIRratio) and offset ratio (BAEI), and the marginal return on a sixth was judged near zero.
The measured 1.8 point gain from all five together supports that judgement.

**Tasselled cap brightness, greenness, wetness.** These are principal-component-like linear combinations that would give a compact orthogonal summary of the six bands.
Excluded because the coefficients are sensor-specific and would have to be sourced and validated for Sentinel-2 MSI, and because tree ensembles do not benefit from orthogonalisation the way linear models do.
Reasonable to revisit if feature count ever becomes the binding constraint.

### 9.3 GLCM texture features not used

Earth Engine's `glcmTexture()` emits 18 metrics per input band.
Six are used from B8 and three from VV, so 27 of a possible 36 texture bands are discarded.

The reason for the cut is dimensional balance.
Taking all 18 from both sources would give 36 texture features against 21 of everything else, so texture would constitute 63% of the feature space and would dominate the RBF kernel's distance calculation outright.
Earth Engine compute cost also scales with band count at both `sampleRegions` and `classify` time, and the pipeline already runs interactively against a request timeout.

The six optical metrics were chosen to span the three Haralick groups rather than to maximise individual performance: contrast group (contrast, dissimilarity, IDM), orderliness group (entropy, ASM), and statistics group (variance).

**Correlation, `g_corr`, is the significant omission.**
The reference paper (Arora et al., 2019) tested all eight standard texture features individually with SVM and found correlation the best performer, followed by mean and variance, and its single best result overall was spectral bands plus NDVI plus correlation at 3x3, Kappa 0.794.
Correlation measures whether brightness varies in a linear, directional way across the neighbourhood, which picks up the aligned repeating structure of streets and building rows.
Nothing in the current six captures directionality.
It is available from the same `glcmTexture()` call at no extra compute and should be added.

**Mean, `g_savg`, sum average.** Ranked second in the reference paper.
Excluded here because a local mean of B8 is close to a smoothed copy of B8, and B8 is already an input, so it is more nearly redundant in this stack than in the paper's.

**Sum variance, sum entropy, difference variance, difference entropy.** Marginal-distribution variants of features already present. Mutually correlated to the point of near-duplication.

**Information measures of correlation 1 and 2, and maximum correlation coefficient.** Entropy-normalised correlation variants.
Numerically unstable when the co-occurrence matrix is sparse, which at 32 grey levels over a small window it frequently is.

**Inertia, cluster shade, cluster prominence.** Higher-order moments.
Extremely sensitive to quantization and to outliers, and near-impossible to interpret, so they fail the "would we understand a change in this number" test.

### 9.4 Texture computed on one band only

Optical texture comes from B8 alone and radar texture from VV alone.
Texture could be computed per band: six optical bands times six metrics is 36 rather than 6.

B8 was chosen because it has the widest dynamic range across all four classes and is a native 10 m band, so the texture is real rather than an artefact of resampling a 20 m band onto a 10 m grid.
Computing texture on upsampled B11 or B12 would measure the interpolator as much as the ground.
The cost of the choice is that texture only ever sees NIR structure, and a surface that is smooth in NIR but rough in SWIR is invisible to this stack.

### 9.5 Radar features not used

**Temporal statistics.** The pipeline takes a median over the date window and discards the temporal dimension entirely.
Standard deviation of VV over a year is a strong built-up feature, because buildings are radiometrically stable while crops and bare fields change dramatically through a season.
This is probably the highest-value unused feature available, and it is not included because the current architecture composites to a single image before sampling.
Adding it is an architectural change, not a one-line change.

**Ascending and descending orbits kept separate.** Double bounce depends on the angle between the radar look direction and the building wall, so a wall aligned with the orbit track responds very differently from a perpendicular one.
Merging both orbit directions into one median averages that away and loses the geometric signal.
Kept merged for scene availability, since splitting halves the number of scenes per composite and the fallback at `gee_classifier.py:113-122` already exists because scenes are sometimes scarce.

**Incidence angle normalisation.** Backscatter varies systematically with the angle the pulse strikes the ground, which within a single Sentinel-1 swath ranges roughly 29 to 46 degrees.
No correction is applied, so part of the VV and VH variation across a wide AOI is viewing geometry rather than land cover.
The VVVH ratio partly cancels this, which is a further argument for that band.

**Terrain flattening.** Radiometric terrain correction for slope is not applied.
On the flat central Indian plain this matters little; over the Satpura and Vindhya ranges it introduces real error.

**Explicit speckle filtering.** No Lee, Refined Lee or Gamma MAP filter is applied.
The temporal median acts as a crude multi-look substitute and costs nothing.
An explicit filter would improve the radar texture bands specifically, since they are the ones speckle corrupts.

**RVI, radar vegetation index, `4*VH / (VV + VH)`.** Excluded as a monotone rearrangement of the VV and VH pair already present, offering nothing a tree split cannot already reach.

**Polarimetric decomposition.** Not available. The GRD product is detected intensity with the phase discarded, and Sentinel-1 is dual-pol rather than fully polarimetric, so entropy-alpha and similar decompositions are impossible with this data source.

### 9.6 Data sources not used at all

**Elevation and slope, from SRTM or Copernicus DEM.** Slope would help, because buildings sit on flat ground and bare rock frequently does not, and it is free and static.
Not included, and a defensible addition.

**Nighttime lights, VIIRS.** An almost unambiguous built-up indicator, and immune to every confusion described in this document.
Excluded on resolution: 500 m against a 10 m classification grid, which would paint entire villages and their surrounding fields with one value.
Useful as a regional prior, not as a per-pixel feature.

**Multi-season composites.** One March-April window is used.
Dry deciduous forest is leaf-off then, which the MP study identifies as the direct cause of Forest recall sitting at 57-68%.
A leaf-on window, or a two-season stack, would address it at the cost of more cloud and roughly double the bands.

**Sentinel-2 temporal standard deviation.** The optical analogue of the radar temporal statistics above, and it would separate cropland from bare soil, currently the single largest unresolvable error in the model.
Same architectural blocker.

**OpenStreetMap building footprints, or any ancillary vector layer.** Excluded on principle: the project's purpose is to classify from imagery, and training or evaluating against an existing built-up map would make the result a copy of that map rather than an independent measurement.

---

## 10. Cross-cutting tradeoffs

**Dimensionality against training data.** 27 features against 5,000 labelled points.
That ratio is acceptable for tree ensembles, which do their own feature selection, and marginal for KNN with k=5, where distance concentration in high dimensions makes all neighbours roughly equidistant.
KNN is the weakest model in the MP study and this is part of why.

**Collinearity.** Many features are algebraic functions of the same six raw bands, so the effective dimensionality is well below 27.
Tree ensembles tolerate this but their feature-importance scores become unreliable, since importance is split arbitrarily among correlated features.
Any future importance-based pruning has to account for that or it will delete the wrong bands.

**Scale mismatch.** B11 and B12 are 20 m natively and everything derived from them inherits that, so 11 of the 27 features are effectively 20 m despite being sampled on a 10 m grid.
Sentinel-1 GRD IW is 10 m pixel spacing but roughly 20 m true resolution.
The nominal 10 m output is optimistic.

**Numeric range.** Normalized differences sit in -1 to +1, radar bands are clamped to 0-1, and SWIRratio and BAEI are unbounded ratios.
Tree models are invariant to this; distance-based models are not.
The SAR rescaling at `gee_classifier.py:126-128` exists specifically to protect the SVM kernel, but SWIRratio and BAEI received no equivalent treatment, which is an inconsistency.

**Compute budget.** Band count multiplies cost at `sampleRegions` (`gee_classifier.py:198-207`) and at `classify` (`:261`), both of which run inside an interactive request.
This is the hard ceiling on how many features can be added, and it is why the exclusions above are decisions rather than oversights.

**Second satellite dependency.** Adding Sentinel-1 introduced orbit, incidence angle and speckle as new failure modes, plus a date-widening fallback path at `gee_classifier.py:113-122` that is not covered by any test.
It bought 5.3 points, which justifies it, but the operational surface area grew.

---

## 11. Known discrepancies with the literature

Two places where the implementation contradicts the reference paper's measured findings.

**Window size.** Arora et al. tested GLCM windows at 3x3, 5x5, 7x7 and 9x9 and found SVM accuracy falling monotonically as the window grew, then fixed on 3x3 for the remainder of their study.
Earth Engine's `glcmTexture(size=N)` uses a (2N+1) by (2N+1) window, so `size=3` at `gee_classifier.py:91` and `:131` is a 7x7 window, two steps into the range they measured as worse.
`size=1` is the paper's 3x3.
This is a one-character change on each line and is worth testing directly against the MP evaluation, since a larger window also blurs small structures and could be part of why isolated rural buildings are missed.

**Correlation is absent.** As set out in section 9.3, correlation was the paper's best individual texture feature and it is not in the stack, while `g_var`, which they ranked worst under MLC, is.
Adding `g_corr` costs nothing at the `glcmTexture()` call.

One place where the paper's finding should explicitly not be followed: it reported that BSI reduced accuracy in every combination.
Its imagery was four-band Planet with no SWIR, so its BSI was a different quantity from the SWIR-based index at `gee_classifier.py:77`.
That result does not transfer and BSI should stay.

---

## 12. Ranked candidate changes

Ordered by expected benefit against effort.

1. **Add `g_corr` to the optical texture selection** at `gee_classifier.py:92`. One token. The reference paper's best individual texture feature, and directionality is currently unrepresented in the stack.
2. **Change `size=3` to `size=1`** at `gee_classifier.py:91` and `:131`. Two characters. Aligns with the paper's measured optimum and sharpens texture on small structures.
3. **Retune SVM `gamma`** at `gee_classifier.py:216`. It was set for 10 features and is running on 27, which is why SVM lost the top spot and why its Water precision fell to 63%. The UI currently offers a model that is known to misbehave.
4. **Correct `MODEL_METADATA`** at `backend/config.py:53-111`. The advertised 98.7-99.35% figures describe a random split inside the training polygon. Measured across MP the honest range is 61-74%.
5. **Add MNDWI** as a standalone band. Its algebraic form only reaches the model indirectly through IBI, and it is the published fix for exactly the built-up water leakage this project fights.
6. **Add DEM slope.** Free, static, and buildings sit on flat ground where bare rock often does not.
7. **Add temporal standard deviation of VV.** Likely the strongest single unused feature, since built-up is stable through a season and bare fields are not. Requires reworking the composite step, so it is the first item here that is not a small change.
8. **Drop `s_var`, and possibly `g_var` and `g_asm`.** Weakest three on both published evidence and redundancy grounds. Only worth doing if the feature count needs to come down, since none of them is actively harmful.

Items 1 through 4 are all small, and each can be evaluated against the existing MP tile harness without new labelling.
