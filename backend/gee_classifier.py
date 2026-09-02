import hashlib
import io
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
from PIL import Image

# A district at 10 m is ~90 megapixels, just over Pillow's default
# decompression-bomb threshold. The big image is one we build ourselves from
# tiles Earth Engine sent us, so the guard is only firing on our own output.
Image.MAX_IMAGE_PIXELS = None

# Support both `python run_app.py` (root in sys.path) and direct module execution
try:
    from backend.config import (
        EE_PROJECT_ID,
        LAND_COVER_CLASSES,
        CLASS_PALETTE,
        BANDS,
        FEATURE_COLLECTIONS,
        MODEL_METADATA,
        TRAINING_SCHEMA_VERSION,
    )
except ModuleNotFoundError:
    # Fallback: add parent dir to path so relative imports resolve
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from backend.config import (
        EE_PROJECT_ID,
        LAND_COVER_CLASSES,
        CLASS_PALETTE,
        BANDS,
        FEATURE_COLLECTIONS,
        MODEL_METADATA,
        TRAINING_SCHEMA_VERSION,
    )

_EE_INITIALIZED = False

# Training never depends on the drawn AOI -- same points, same composite, same result
# every request -- so the trained classifier is reusable across AOIs.
# ponytail: plain dict, no eviction; five models x a handful of date windows is bounded.
_CLASSIFIER_CACHE = {}


def _classifier_cache_key(model_type, start_date, end_date, cloud_threshold,
                          allow_temporal_fallback):
    return (
        model_type.lower(),
        start_date,
        end_date,
        cloud_threshold,
        # Part of the key because it changes which imagery the model was
        # trained on. Without it a dashboard request and an evaluation run
        # would silently share one cached classifier.
        allow_temporal_fallback,
        TRAINING_SCHEMA_VERSION,
    )


# One Earth Engine constructor per model id. The settings themselves live in
# config.MODEL_METADATA, not here: they used to be written out twice, once in
# this function and once in the metadata the dashboard displays, and two copies
# of a hyperparameter is one copy too many. Retuning now edits a single place
# and the served description, the trained model and the paper agree by
# construction.
CLASSIFIER_FACTORIES = {
    "rf": lambda: ee.Classifier.smileRandomForest,
    "svm": lambda: ee.Classifier.libsvm,
    # "xgb" is the pre-v4 spelling and still resolves, so old notebooks and
    # saved request payloads keep working. The name was wrong:
    # ee.Classifier.smileGradientTreeBoost is Smile's gradient tree boosting,
    # not the XGBoost of Chen and Guestrin, and nothing about this classifier
    # is XGBoost's algorithm.
    "gtb": lambda: ee.Classifier.smileGradientTreeBoost,
    "xgb": lambda: ee.Classifier.smileGradientTreeBoost,
    "cart": lambda: ee.Classifier.smileCart,
    "knn": lambda: ee.Classifier.smileKNN,
}

MODEL_ALIASES = {"xgb": "gtb"}


def classifier_parameters(model_type):
    """The settings one model ships with, from the single place they live."""
    model = MODEL_ALIASES.get(model_type.lower(), model_type.lower())
    if model not in MODEL_METADATA:
        raise ValueError(f"Unsupported model: {model_type}")
    return dict(MODEL_METADATA[model]["params"])


def make_classifier(model_type, params=None):
    """Build one of the five supported classifiers.

    `params` overrides the shipped settings, which is what the hyperparameter
    search on the development districts needs; leaving it None gives exactly
    the configuration the dashboard serves and the paper reports.
    """
    model = model_type.lower()
    factory = CLASSIFIER_FACTORIES.get(model)
    if factory is None:
        raise ValueError(f"Unsupported model: {model_type}")
    settings = classifier_parameters(model) if params is None else dict(params)
    # A None means "leave this argument unset", which is how the search says
    # "no node cap" without Earth Engine seeing a null.
    settings = {key: value for key, value in settings.items()
                if value is not None}
    return factory()(**settings)


def merge_feature_collections(paths):
    """Load and merge asset paths in deterministic order."""
    paths = list(paths)
    if not paths:
        raise ValueError("At least one training collection is required")
    merged = ee.FeatureCollection(paths[0])
    for path in paths[1:]:
        merged = merged.merge(ee.FeatureCollection(path))
    return merged


def init_ee():
    global _EE_INITIALIZED
    if not _EE_INITIALIZED:
        try:
            # Hosted deployments have no browser to run `earthengine authenticate`
            # and no credentials file. The key arrives whole as an env var because
            # secret stores (HF Spaces, Render) expose secrets as env vars, not
            # files, so pass it through as key_data rather than staging a tempfile.
            key_data = os.getenv("EE_SERVICE_ACCOUNT_KEY")
            if key_data:
                email = json.loads(key_data)["client_email"]
                ee.Initialize(
                    ee.ServiceAccountCredentials(email, key_data=key_data),
                    project=EE_PROJECT_ID,
                )
            else:
                ee.Initialize(project=EE_PROJECT_ID)
            _EE_INITIALIZED = True
            print(f"Google Earth Engine initialized successfully with project '{EE_PROJECT_ID}'")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            raise e
    return _EE_INITIALIZED


def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


# Bands that are not already 0-1. The GLCM moments are the reason this exists:
# raw g_var reaches 143 and s_var 98, so those two plus g_contrast/s_contrast
# were 97% of the Euclidean distance that smileKNN and the libsvm RBF kernel
# operate on, drowning every optical and radar band; g_var alone was 51%. After
# rescaling those four are 21% and no band exceeds 16%. SWIRratio and BAEI are
# ratios with no upper bound at all, so the clamp is protection as much as
# scaling. Trees split on thresholds and are almost unaffected -- they moved by
# at most 0.4 points, which is the clamp tying the top 1% together and removing
# the few splits that lived inside that tail.
#
# Bounds are the winter p99 of the labelled training samples, rounded up
# (winter has the wider tails). They must come from that class-stratified
# sample and not from a district-wide pixel sweep: districts are mostly
# vegetation and cropland, so a district p99 badly understates a built-up index
# on the class-balanced set the classifier actually sees. BAEI measured 2.7 on
# districts against 8.1 on the training samples, and a bound of 3.0 clamped
# 12% of winter rows to a single tied value -- cutting into the body of the
# distribution rather than trimming a tail. Bounds that are too wide are the
# opposite failure and just as real: they compress a band into a fraction of
# the scale and quietly under-weight it in the same distance metric this table
# exists to balance. Verify with the saturation check before trusting a bound:
# every band should sit near 1% saturated, neither 12% nor 0%.
FEATURE_RANGES = {
    "SWIRratio": (0.0, 2.5),
    "BAEI": (0.0, 9.0),
    "g_contrast": (0.0, 22.0),
    "g_ent": (0.0, 4.2),
    "g_var": (0.0, 42.0),
    "g_diss": (0.0, 3.6),
    "s_contrast": (0.0, 13.5),
    "s_var": (0.0, 34.0),
    "s_ent": (0.0, 4.0),
}


def _unit_range(image, name):
    """Map one band onto 0-1 using its entry in FEATURE_RANGES."""
    low, high = FEATURE_RANGES[name]
    return image.select(name).unitScale(low, high).clamp(0, 1).rename(name)


def _rescale_bands(image, names):
    for name in names:
        image = image.addBands(_unit_range(image, name), overwrite=True)
    return image


def _add_spectral_indices(composite):
    """Add index bands plus GLCM texture to a composite.

    Bare soil and built-up overlap almost completely in per-pixel reflectance, so
    the separation has to come from somewhere else: bareness/built indices that
    exploit the SWIR slope, and texture, which measures local structure rather
    than brightness.
    """
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    savi = composite.expression(
        "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        {"NIR": composite.select("B8"), "RED": composite.select("B4")}
    ).rename("SAVI")
    composite = composite.addBands([ndvi, ndwi, ndbi, savi])

    b2, b3, b4 = composite.select("B2"), composite.select("B3"), composite.select("B4")
    b8, b11, b12 = composite.select("B8"), composite.select("B11"), composite.select("B12")

    bsi = (b11.add(b4).subtract(b8.add(b2))) \
        .divide(b11.add(b4).add(b8).add(b2)).rename("BSI")
    ui = (b12.subtract(b8)).divide(b12.add(b8)).rename("UI")
    t1 = b11.multiply(2).divide(b11.add(b8))
    t2 = b8.divide(b8.add(b4)).add(b3.divide(b3.add(b11)))
    ibi = (t1.subtract(t2)).divide(t1.add(t2)).rename("IBI")
    swir_ratio = b11.divide(b12.add(1e-6)).rename("SWIRratio")
    baei = (b4.add(0.3)).divide(b3.add(b11).add(1e-6)).rename("BAEI")
    composite = composite.addBands([bsi, ui, ibi, swir_ratio, baei])
    composite = _rescale_bands(composite, ("SWIRratio", "BAEI"))

    # GLCM needs a small number of grey levels; 32 keeps the co-occurrence matrix
    # dense enough that entropy and homogeneity stay meaningful.
    grey = composite.select("B8").unitScale(0, 0.5).clamp(0, 1) \
        .multiply(31).toByte().rename("g")
    texture = grey.glcmTexture(size=3).select(
        ["g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm"])
    texture = _rescale_bands(texture, ("g_contrast", "g_ent", "g_var", "g_diss"))
    return composite.addBands(texture)


# One orbit direction only. Sentinel-1 backscatter depends on the viewing
# geometry as much as on the ground: ascending and descending passes see the
# same slope, wall or furrow from opposite sides, so mixing them into one median
# adds a look-direction term to every radar feature that the classifier can only
# read as noise. Descending is the pass with the denser Indian archive.
S1_ORBIT_PASS = "DESCENDING"


def sentinel1_collection(geometry, start_date, end_date):
    """The canonical Sentinel-1 selection: IW, dual-pol, one orbit direction."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT_PASS))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def _add_sar(composite, geometry, start_date, end_date,
             allow_temporal_fallback=False):
    """Add Sentinel-1 C-band radar bands.

    Buildings produce a double-bounce return that bare soil cannot, at any
    brightness, so radar separates the two by physical mechanism rather than by
    appearance. Values are rescaled from dB into roughly 0-1 so they sit on the
    same scale as reflectance -- an RBF SVM is distance-based and unscaled dB
    (-25..5) would otherwise dominate the kernel.

    `allow_temporal_fallback` is the difference between the dashboard and the
    experiments. A user who draws a polygon over a week with no radar pass wants
    a map, so the dashboard widens to the 2024-2025 archive rather than handing
    back a blank. An accuracy figure cannot afford that: a 2024 monsoon scene
    standing in for an April 2025 one describes a different season and a
    different ground state, and the seasonal comparison the paper reports would
    then be partly a comparison of fallback rates. With the fallback off, an
    empty window yields fully masked radar bands, the affected points fail the
    notNull filter, and the missingness is visible in the sample count instead
    of being silently papered over.
    """
    s1 = sentinel1_collection(geometry, start_date, end_date)
    if allow_temporal_fallback:
        s1 = ee.ImageCollection(ee.Algorithms.If(
            s1.size().gt(0),
            s1,
            sentinel1_collection(geometry, "2024-01-01", "2025-12-31"),
        ))
        sar = s1.select(["VV", "VH"]).median()
    else:
        # median() of an empty collection has no bands at all, so select("VV")
        # would raise rather than mask. Substitute a fully masked stand-in that
        # keeps the band names and lets the notNull filter drop the points.
        blank = ee.Image.constant([0, 0]).rename(["VV", "VH"]) \
            .updateMask(ee.Image.constant(0)).toFloat()
        sar = ee.Image(ee.Algorithms.If(
            s1.size().gt(0), s1.select(["VV", "VH"]).median(), blank))
    return add_sar_bands(composite, sar)


def add_sar_bands(composite, sar):
    """Append the rescaled radar amplitude and texture bands from a VV/VH median.

    Split out from _add_sar so scripts/band_ablation.py, which selects its own
    Sentinel-1 collection to avoid _add_sar's per-tile .getInfo() probes, shares
    this band maths instead of copying it. It previously copied it, and the copy
    silently missed the texture rescaling.
    """
    vv_db, vh_db = sar.select("VV"), sar.select("VH")
    # The polarimetric ratio is taken as a *difference of decibels*, which is
    # identically log10(sigma0_VV / sigma0_VH) up to the 10x factor -- the same
    # physical quantity a linear-power ratio measures. Dividing the dB values,
    # or dividing after the 0-1 rescaling below, would not be: neither is a
    # backscatter ratio in any units. The rescale is applied only afterwards,
    # to the finished ratio, purely so the distance-based classifiers see it on
    # the same footing as reflectance.
    ratio = vv_db.subtract(vh_db).unitScale(-5, 20).clamp(0, 1).rename("VVVH")
    vv = vv_db.unitScale(-25, 5).clamp(0, 1).rename("VV")
    vh = vh_db.unitScale(-30, 0).clamp(0, 1).rename("VH")

    grey = vv_db.unitScale(-25, 5).clamp(0, 1).multiply(31).toByte().rename("s")
    texture = grey.glcmTexture(size=3).select(["s_contrast", "s_var", "s_ent"])
    texture = _rescale_bands(texture, ("s_contrast", "s_var", "s_ent"))
    return composite.addBands([vv, vh, ratio]).addBands(texture)


def _build_collection(geometry, start_date, end_date, cloud_threshold,
                      allow_temporal_fallback=False):
    """Build a filtered & cloud-masked Sentinel-2 image collection for a geometry.

    See `_add_sar` for why `allow_temporal_fallback` exists: the same
    dashboard-versus-experiment split applies to the optical archive.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
    )
    if not allow_temporal_fallback:
        return collection
    fallback = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2024-01-01", "2025-12-31")
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_s2_clouds)
    )
    return ee.ImageCollection(
        ee.Algorithms.If(collection.size().gt(0), collection, fallback)
    )


def build_sentinel_composite(geometry, start_date="2025-04-01", end_date="2025-05-30",
                             cloud_threshold=15, allow_temporal_fallback=False):
    """Build a Sentinel-2 median composite clipped to `geometry` with spectral indices."""
    init_ee()
    collection = _build_collection(
        geometry, start_date, end_date, cloud_threshold, allow_temporal_fallback)
    composite = collection.median().clip(geometry)
    composite = _add_spectral_indices(composite)
    composite = _add_sar(composite, geometry, start_date, end_date,
                         allow_temporal_fallback)
    return composite


def sample_training_points(
    points,
    start_date="2025-04-01",
    end_date="2025-05-30",
    cloud_threshold=15,
    allow_temporal_fallback=False,
    bands=None,
):
    """Sample a feature stack for a labelled point collection.

    `bands` defaults to the shipped 19-band cut. Pass config.FULL_BANDS to get
    the 27-band superset, which is what the feature-stack comparison needs: one
    sampled table that every candidate stack can be trained from, so the stacks
    differ only in which columns the classifier is allowed to see and not in
    which pixels survived sampling.
    """
    train_geometry = points.geometry().bounds()
    train_collection = _build_collection(
        train_geometry, start_date, end_date, cloud_threshold,
        allow_temporal_fallback)
    train_composite = _add_spectral_indices(train_collection.median())
    train_composite = _add_sar(
        train_composite, train_geometry, start_date, end_date,
        allow_temporal_fallback)
    bands = list(BANDS if bands is None else bands)
    return _sample_regions_with_geometry(
        train_composite.select(bands), points, bands)


def _sample_regions_with_geometry(image, points, bands=None):
    """Sample labelled points into an exportable table with point geometries."""
    bands = list(BANDS if bands is None else bands)
    return (
        image.sampleRegions(
            collection=points,
            properties=["label"],
            scale=10,
            tileScale=4,
            geometries=True,
        )
        .filter(ee.Filter.notNull(bands + ["label"]))
    )


def get_trained_classifier(model_type, start_date="2025-04-01", end_date="2025-05-30",
                           cloud_threshold=15, allow_temporal_fallback=False):
    """
    ALWAYS train on the composite covering the labelled training points.
    The classifier is then applied to any user-supplied AOI.
    """
    init_ee()

    cache_key = _classifier_cache_key(
        model_type, start_date, end_date, cloud_threshold,
        allow_temporal_fallback)
    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    # ── Load training label feature collections ──
    # Merged generically so adding a collection to FEATURE_COLLECTIONS is enough;
    # every asset carries its own `label`, so class identity travels with the points.
    all_points = merge_feature_collections(FEATURE_COLLECTIONS.values())
    training_samples = sample_training_points(
        all_points,
        start_date=start_date,
        end_date=end_date,
        cloud_threshold=cloud_threshold,
        allow_temporal_fallback=allow_temporal_fallback,
    )

    # ── Instantiate the requested classifier ──
    classifier = make_classifier(model_type)

    trained_classifier = classifier.train(
        features=training_samples,
        classProperty="label",
        inputProperties=BANDS
    )
    _CLASSIFIER_CACHE[cache_key] = trained_classifier
    return trained_classifier


def _iter_coords(coords):
    """Yield (lon, lat) pairs from an arbitrarily nested GeoJSON coordinate list."""
    if coords and isinstance(coords[0], (int, float)):
        yield coords[0], coords[1]
        return
    for part in coords:
        yield from _iter_coords(part)


def _static_overlay(image, user_geometry, geometry_dict, vis_params, max_px=2048):
    """Render an AOI layer once, as a single image.

    True-colour only. Downsampling reflectance is meaningful -- the average of
    four green pixels is still green -- but downsampling a *classification* is
    not, and asking Earth Engine for one at display scale silently re-runs the
    model there. See the note above `render_overlay_png`.

    GEE's XYZ tile endpoints are *live inference*: every 256px tile the map asks
    for re-runs the whole composite -> SAR -> texture -> classify chain on GEE.
    A viewport needs dozens of them, so the map paints in one square at a time
    and pays the whole bill again on every pan and zoom.

    An AOI is bounded, though, and Sentinel-2 is 10 m -- a 5 km box is only
    ~500 px of real data. So render it once and hand the client a flat image.
    EPSG:3857 so the result drops onto a Web-Mercator map as a plain rectangle
    with no reprojection, and PNG so everything outside the AOI stays
    transparent.
    """
    # Computed locally: the AOI came from the client as plain GeoJSON, so asking
    # Earth Engine for its own bounding box would be a network round trip to
    # learn something already sitting in memory.
    pts = list(_iter_coords(geometry_dict["coordinates"]))
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    # Ask for native 10 m resolution, but never more pixels than max_px --
    # beyond that the image costs more to ship than the detail is worth.
    width_m = (east - west) * 111320.0 * math.cos(math.radians((north + south) / 2))
    px = max(256, min(max_px, int(width_m / 10)))

    url = ""
    try:
        url = image.getThumbURL({
            "region": user_geometry,
            "dimensions": px,
            "crs": "EPSG:3857",
            "format": "png",
            **vis_params,
        })
    except Exception as ex:
        print(f"Static overlay generation notice: {ex}")

    return {
        "url": url,
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "width_px": px,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Native-resolution overlay rendering
#
# The classifier is only valid at the scale it was trained at. Training points
# were sampled with `scale=10`, and two of the nineteen features are GLCM
# textures over a 3x3 *pixel* window -- so "3x3" means 30 m on the training grid
# and 180 m on a 60 m grid. Ask Earth Engine for the classified image at any
# other scale and the whole chain (composite -> indices -> texture -> classify)
# is re-evaluated there, on mean-pyramided reflectance, and the model answers a
# question it was never trained on.
#
# Measured on Jabalpur district, same request, water as a share of the AOI:
#
#     scale     10 m    30 m    50 m    100 m
#     water    5.65%  31.99%  28.54%   26.52%
#
# That is not a rendering artefact -- every pixel came back an exact palette
# colour, so nothing was blended. It is a different classification. The old
# renderer capped the thumbnail at 2048 px, which for a district is ~60 m/px,
# so the picture on the map disagreed with the numbers beside it by 6x on
# Water, and the numbers were the trustworthy half.
#
# So the overlay is rendered at the training scale and nowhere else. Earth
# Engine will not do that in one request -- a district on a 10 m grid is
# ~13700x7700 px and it answers "Reprojection output too large" -- so the AOI is
# cut into tiles, each rendered at 10 m, and stitched back together here. It is
# slow (~2 min per tile, and EE starts returning 429 above three at a time), so
# it runs once in the background and the result is cached on disk: the frontend
# gets a flat PNG it can pan and zoom over for free, exactly as before.
# ─────────────────────────────────────────────────────────────────────────────

OVERLAY_CACHE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".overlay_cache"))

# One tile is ~2.6 MP of inference and takes ~2 min. Bigger tiles mean fewer
# round trips but EE fails the whole tile on a memory limit somewhere above
# ~4 MP, and a failed tile is a hole in the map.
TILE_PX = 1792

# Measured ceiling: EE returns 429 on the fourth concurrent thumbnail. Projects
# in Restricted Mode share that budget with everything else the app is doing, so
# a render at full tilt will starve incoming classify requests -- drop this to 1
# on a constrained tier.
RENDER_WORKERS = int(os.getenv("OVERLAY_RENDER_WORKERS", "3"))

_EARTH_RADIUS_M = 6378137.0

# GLCM uses a 3x3 window and focalMode another; 24 px of overlap is far more
# than either needs and costs nothing, the padding is cropped away.
_EDGE_PAD_PX = 24

# ponytail: plain dict + lock. One entry per rendered AOI, a handful per session.
_RENDER_JOBS = {}
_RENDER_LOCK = threading.Lock()


def _merc_x(lon):
    return math.radians(lon) * _EARTH_RADIUS_M


def _merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * _EARTH_RADIUS_M


def _inv_merc_x(x):
    return math.degrees(x / _EARTH_RADIUS_M)


def _inv_merc_y(y):
    return math.degrees(2 * math.atan(math.exp(y / _EARTH_RADIUS_M)) - math.pi / 2)


def _aoi_bounds(geometry_dict):
    """Bounding box of a GeoJSON geometry, computed locally.

    The AOI arrived from the client as plain GeoJSON, so asking Earth Engine for
    its own bounding box would be a network round trip to learn something
    already sitting in memory.
    """
    pts = list(_iter_coords(geometry_dict["coordinates"]))
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return min(lons), min(lats), max(lons), max(lats)


def overlay_cache_key(geometry_dict, model_type, start_date, end_date,
                      cloud_threshold, smoothing):
    """Stable id for one rendered overlay.

    Everything that changes a pixel is in the key, the schema version included,
    so a cached PNG from an older class inventory can never be served against
    new numbers.
    """
    payload = json.dumps({
        "geometry": geometry_dict,
        "model": model_type.lower(),
        "start": start_date,
        "end": end_date,
        "cloud": cloud_threshold,
        "smoothing": bool(smoothing),
        "schema": TRAINING_SCHEMA_VERSION,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def overlay_cache_path(key):
    return os.path.join(OVERLAY_CACHE_DIR, f"{key}.png")


def _render_grid(geometry_dict, scale_m=10.0):
    """Pixel grid for the AOI at `scale_m` ground metres, in Web Mercator.

    Web Mercator metres are inflated by 1/cos(lat), so a request for "10 m" in
    EPSG:3857 buys ~9.2 ground metres at Jabalpur's latitude. The grid is built
    the other way round -- pick the mercator scale that lands on 10 ground
    metres -- so the render sits on the same grid the model was trained on
    rather than merely near it.
    """
    west, south, east, north = _aoi_bounds(geometry_dict)
    lat_mid = (north + south) / 2.0
    merc_scale = scale_m / math.cos(math.radians(lat_mid))

    x0, x1 = _merc_x(west), _merc_x(east)
    y0, y1 = _merc_y(south), _merc_y(north)
    width = max(1, int(round((x1 - x0) / merc_scale)))
    height = max(1, int(round((y1 - y0) / merc_scale)))

    # A whole number of pixels rarely lands exactly on the AOI's corner, so the
    # image covers a hair more ground than the bbox asked for. Report where the
    # pixels actually are, anchored at the top-left: hand Leaflet the requested
    # bbox instead and the overlay sits a fraction of a pixel off true.
    return {
        "west": west,
        "north": north,
        "east": _inv_merc_x(x0 + width * merc_scale),
        "south": _inv_merc_y(y1 - height * merc_scale),
        "x0": x0, "y1": y1, "merc_scale": merc_scale,
        "width": width, "height": height,
    }


def _tile_boxes(grid):
    """Split the grid into tile-sized pixel boxes, each with its lon/lat region.

    Tiles are cut on the mercator pixel grid, not on latitude, so every tile is
    exactly TILE_PX rows tall and the pieces butt together without a seam.
    """
    boxes = []
    for top in range(0, grid["height"], TILE_PX):
        for left in range(0, grid["width"], TILE_PX):
            right = min(left + TILE_PX, grid["width"])
            bottom = min(top + TILE_PX, grid["height"])
            s = grid["merc_scale"]
            boxes.append({
                "px": (left, top, right, bottom),
                "region": [
                    _inv_merc_x(grid["x0"] + left * s),
                    _inv_merc_y(grid["y1"] - bottom * s),
                    _inv_merc_x(grid["x0"] + right * s),
                    _inv_merc_y(grid["y1"] - top * s),
                ],
            })
    return boxes


def _is_rate_limited(ex):
    """Is this Earth Engine saying "slow down" rather than "no"?

    The two calls in a tile fail differently: `getThumbURL` goes through the EE
    API and raises `EEException`, the download that follows raises
    `urllib.HTTPError`. Both report a 429, and neither is fatal -- a render is
    half an hour of work, so anything that might just be congestion is worth
    waiting out rather than discarding the whole job over.
    """
    if isinstance(ex, urllib.error.HTTPError):
        # 429 is EE's own rate limiter; the 5xx family is its fronting
        # infrastructure being briefly unavailable. Both are transient and both
        # cost the whole render if treated as fatal -- a measured Etawah run
        # lost 9 completed tiles and 10 minutes to a single 503 on tile 10.
        return ex.code == 429 or ex.code in (500, 502, 503, 504)
    text = str(ex).lower()
    return ("too many requests" in text or "concurrency limit" in text
            or "service unavailable" in text or "backend error" in text)


def _fetch_tile(image_fn, box, attempts=6):
    """Render one tile at the training scale, retrying EE's rate limiter.

    429 is the expected steady state, not an error: three concurrent renders is
    already EE's ceiling, so a fourth in flight -- from another request, or a
    retry -- backs off rather than failing the tile and punching a hole in the
    map.
    """
    left, top, right, bottom = box["px"]
    want = (right - left, bottom - top)
    west, south, east, north = box["region"]
    region = ee.Geometry.Rectangle(
        [west, south, east, north], proj="EPSG:4326", geodesic=False)

    # Texture and focalMode read a neighbourhood, so a composite cut to the tile
    # edge has nothing to read there and every seam comes back misclassified.
    # Build on a padded region, then crop back to the exact tile.
    pad_x = (east - west) * _EDGE_PAD_PX / max(1, want[0])
    pad_y = (north - south) * _EDGE_PAD_PX / max(1, want[1])
    padded = ee.Geometry.Rectangle(
        [west - pad_x, south - pad_y, east + pad_x, north + pad_y],
        proj="EPSG:4326", geodesic=False)

    for attempt in range(attempts):
        try:
            url = image_fn(padded).getThumbURL({
                "region": region,
                "dimensions": f"{want[0]}x{want[1]}",
                "crs": "EPSG:3857",
                "format": "png",
                "min": 0,
                "max": len(CLASS_PALETTE) - 1,
                "palette": CLASS_PALETTE,
            })
            data = urllib.request.urlopen(url, timeout=600).read()
            tile = Image.open(io.BytesIO(data)).convert("RGBA")
            if tile.size != want:
                tile = tile.resize(want, Image.NEAREST)
            return tile
        except Exception as ex:
            if attempt == attempts - 1 or not _is_rate_limited(ex):
                raise
            # Backing off further than EE needs is free; the alternative is
            # losing every tile rendered so far.
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError("tile render exhausted retries")


def render_overlay_png(image_fn, geometry_dict, key, progress=None):
    """Render the AOI at the training scale, tile by tile, and cache the PNG."""
    grid = _render_grid(geometry_dict)
    boxes = _tile_boxes(grid)
    canvas = Image.new("RGBA", (grid["width"], grid["height"]), (0, 0, 0, 0))

    done = 0
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        futures = {pool.submit(_fetch_tile, image_fn, b): b for b in boxes}
        # As they land, not in submission order: tiles take uneven times, and a
        # progress readout that only moves when the *next* tile finishes looks
        # stuck for minutes at a stretch.
        for future in as_completed(futures):
            canvas.paste(future.result(), futures[future]["px"][:2])
            done += 1
            if progress:
                progress(done, len(boxes))

    os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)
    path = overlay_cache_path(key)
    # Write beside the target and rename: a reader polling the cache must never
    # see a half-written PNG.
    tmp_path = f"{path}.{os.getpid()}.part"
    try:
        canvas.save(tmp_path, "PNG", optimize=True)
        os.replace(tmp_path, path)
    except BaseException:
        # Including the interpreter shutting down mid-save: the render runs on a
        # daemon thread, so a part file outliving it would sit in the directory
        # the overlays are served from forever.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path, grid


def overlay_status(key):
    """Where one overlay render has got to."""
    if os.path.exists(overlay_cache_path(key)):
        return {"status": "ready", "url": f"/overlays/{key}.png"}
    with _RENDER_LOCK:
        job = _RENDER_JOBS.get(key)
    if not job:
        return {"status": "unknown"}
    return dict(job)


def start_overlay_render(image_fn, geometry_dict, key):
    """Kick off (or join) the background render for one AOI.

    Rendering a district at 10 m is ~25 minutes of Earth Engine compute, which
    no HTTP request can wait for, so it runs on a thread and the frontend polls.
    The cached PNG is the whole point: it is paid once per AOI and then panned
    and zoomed over for free.
    """
    if os.path.exists(overlay_cache_path(key)):
        return {"status": "ready", "url": f"/overlays/{key}.png"}

    with _RENDER_LOCK:
        if key in _RENDER_JOBS:
            return dict(_RENDER_JOBS[key])
        grid = _render_grid(geometry_dict)
        total = len(_tile_boxes(grid))
        _RENDER_JOBS[key] = {
            "status": "rendering", "url": "", "done": 0, "total": total,
        }

    def _progress(done, total_tiles):
        with _RENDER_LOCK:
            _RENDER_JOBS[key].update(done=done, total=total_tiles)

    def _run():
        try:
            render_overlay_png(image_fn, geometry_dict, key, _progress)
            with _RENDER_LOCK:
                _RENDER_JOBS[key].update(
                    status="ready", url=f"/overlays/{key}.png")
        except Exception as ex:
            print(f"Overlay render failed for {key}: {ex}")
            with _RENDER_LOCK:
                _RENDER_JOBS[key].update(status="failed", error=str(ex))

    threading.Thread(target=_run, name=f"overlay-{key}", daemon=True).start()
    with _RENDER_LOCK:
        return dict(_RENDER_JOBS[key])


def classify_and_analyze(
    geometry_dict,
    model_type="rf",
    start_date="2025-04-01",
    end_date="2025-05-30",
    cloud_threshold=15,
    smoothing=True,
    allow_temporal_fallback=True,
):
    """Classify a user-drawn AOI. This is the dashboard path, and the only one
    that widens to the 2024-2025 archive when the requested window is empty:
    somebody who drew a polygon wants a map back, whereas a reported accuracy
    figure cannot be allowed to quietly describe a different year. Every
    evaluation script therefore leaves `allow_temporal_fallback` at its False
    default; see `_add_sar`.
    """
    start_time = time.time()
    init_ee()

    # ── 1. Train on the labelled-point composite (training labels always live there) ──
    classifier = get_trained_classifier(
        model_type, start_date, end_date, cloud_threshold,
        allow_temporal_fallback)

    # ── 2. Build the user's AOI composite for classification ──
    user_geometry = ee.Geometry(geometry_dict)
    aoi_composite = build_sentinel_composite(
        geometry=user_geometry,
        start_date=start_date,
        end_date=end_date,
        cloud_threshold=cloud_threshold,
        allow_temporal_fallback=allow_temporal_fallback,
    )

    # ── 3. Classify the user's AOI ──
    def _classify(image, clip_to=None):
        out = image.select(BANDS).classify(classifier)
        if smoothing:
            out = out.focalMode(radius=1, kernelType="square", units="pixels")
        return out.clip(clip_to) if clip_to is not None else out

    classified = _classify(aoi_composite)

    # The tiled renderer rebuilds the composite per tile rather than reusing the
    # AOI-wide one: a tile is a different `filterBounds`, and the point is to
    # keep each request small enough that Earth Engine will evaluate it on the
    # 10 m grid at all. Clipped back to the AOI so the district keeps its shape
    # and everything outside it stays transparent.
    def _tile_image(region):
        return _classify(
            build_sentinel_composite(
                geometry=region,
                start_date=start_date,
                end_date=end_date,
                cloud_threshold=cloud_threshold,
                allow_temporal_fallback=allow_temporal_fallback,
            ),
            clip_to=user_geometry,
        )

    overlay_key = overlay_cache_key(
        geometry_dict, model_type, start_date, end_date, cloud_threshold, smoothing)

    # ── 4. Render overlays, class histogram and export URL, in parallel ──
    # Each of these is an independent synchronous HTTPS round trip to GEE
    # (~1-2s each); run sequentially the API latency is their sum, in parallel
    # it is their max.
    def _histogram():
        return classified.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=user_geometry,
            scale=10,
            maxPixels=1e9
        ).getInfo()

    def _geotiff_url():
        try:
            return classified.getDownloadURL({
                "name":   f"terrain_classified_{model_type}",
                "scale":  10,
                "crs":    "EPSG:4326",
                "region": user_geometry,
                "format": "GEO_TIFF"
            })
        except Exception as ex:
            print(f"GeoTIFF URL generation notice: {ex}")
            return ""

    with ThreadPoolExecutor(max_workers=4) as pool:
        rgb_future = pool.submit(
            _static_overlay, aoi_composite, user_geometry, geometry_dict,
            {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3})
        terrain_future = pool.submit(
            start_overlay_render, _tile_image, geometry_dict, overlay_key)
        histogram_future = pool.submit(_histogram)
        download_future = pool.submit(_geotiff_url)

    rgb_overlay = rgb_future.result()
    grid = _render_grid(geometry_dict)
    overlay = {
        "key": overlay_key,
        "bounds": {
            "west": grid["west"], "south": grid["south"],
            "east": grid["east"], "north": grid["north"],
        },
        "width_px": grid["width"],
        "height_px": grid["height"],
        "scale_m": 10,
        **terrain_future.result(),
    }
    histogram = histogram_future.result()
    download_url = download_future.result()

    # ── 5. Compute individual class surface areas ──

    class_counts = histogram.get("classification", {}) or {}

    PIXEL_AREA_SQM = 100.0   # 10m × 10m Sentinel-2 pixel
    total_pixels   = sum(int(v) for v in class_counts.values()) if class_counts else 0
    total_area_sqm = total_pixels * PIXEL_AREA_SQM
    total_area_ha  = total_area_sqm / 10000.0
    total_area_acres = total_area_sqm / 4046.856

    individual_class_areas = []
    for class_id in LAND_COVER_CLASSES:
        class_info  = LAND_COVER_CLASSES[class_id]
        # GEE may return int or string keys
        pixel_count = int(
            class_counts.get(str(class_id),
            class_counts.get(class_id, 0)) or 0
        )
        sqm  = pixel_count * PIXEL_AREA_SQM
        ha   = sqm / 10000.0
        acres = sqm / 4046.856
        pct  = (sqm / total_area_sqm * 100.0) if total_area_sqm > 0 else 0.0

        individual_class_areas.append({
            "class_id":    class_id,
            "name":        class_info["name"],
            "color":       class_info["color"],
            "description": class_info["description"],
            "pixel_count": pixel_count,
            "area_sqm":    round(sqm, 2),
            "area_ha":     round(ha, 4),
            "area_acres":  round(acres, 4),
            "percentage":  round(pct, 2)
        })

    elapsed_sec = round(time.time() - start_time, 2)
    model_meta  = MODEL_METADATA.get(model_type.lower(), MODEL_METADATA["rf"])

    return {
        "status": "success",
        "model": {
            "id":                model_type,
            "name":              model_meta["name"],
            "type":              model_meta["type"],
            "description":       model_meta["description"],
            "benchmark":         model_meta["benchmark"],
        },
        "rgb_overlay": rgb_overlay,
        "terrain_overlay": overlay,
        "summary": {
            "total_pixels":      total_pixels,
            "total_area_sqm":    round(total_area_sqm, 2),
            "total_area_ha":     round(total_area_ha, 4),
            "total_area_acres":  round(total_area_acres, 4),
            "processing_time_sec": elapsed_sec
        },
        "individual_class_areas": individual_class_areas,
        "export": {
            "geotiff_url": download_url
        }
    }
