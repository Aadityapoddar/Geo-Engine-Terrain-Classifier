import math
import os
import sys
import time
import ee

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


def _classifier_cache_key(model_type, start_date, end_date, cloud_threshold):
    return (
        model_type.lower(),
        start_date,
        end_date,
        cloud_threshold,
        TRAINING_SCHEMA_VERSION,
    )


def make_classifier(model_type):
    """Build one of the five supported classifiers with canonical parameters."""
    model = model_type.lower()
    if model == "rf":
        return ee.Classifier.smileRandomForest(
            numberOfTrees=200,
            minLeafPopulation=1,
            bagFraction=0.3,
            maxNodes=10,
        )
    if model == "svm":
        return ee.Classifier.libsvm(kernelType="RBF", gamma=1.0, cost=100.0)
    if model == "xgb":
        return ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=100,
            shrinkage=0.1,
            maxNodes=10,
        )
    if model == "cart":
        return ee.Classifier.smileCart(maxNodes=20)
    if model == "knn":
        return ee.Classifier.smileKNN(k=5)
    raise ValueError(f"Unsupported model: {model_type}")


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

    # GLCM needs a small number of grey levels; 32 keeps the co-occurrence matrix
    # dense enough that entropy and homogeneity stay meaningful.
    grey = composite.select("B8").unitScale(0, 0.5).clamp(0, 1) \
        .multiply(31).toByte().rename("g")
    texture = grey.glcmTexture(size=3).select(
        ["g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm"])
    return composite.addBands(texture)


def _add_sar(composite, geometry, start_date, end_date):
    """Add Sentinel-1 C-band radar bands.

    Buildings produce a double-bounce return that bare soil cannot, at any
    brightness, so radar separates the two by physical mechanism rather than by
    appearance. Values are rescaled from dB into roughly 0-1 so they sit on the
    same scale as reflectance -- an RBF SVM is distance-based and unscaled dB
    (-25..5) would otherwise dominate the kernel.
    """
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    sar_fallback = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate("2024-01-01", "2025-12-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    s1 = ee.ImageCollection(
        ee.Algorithms.If(s1.size().gt(0), s1, sar_fallback)
    )

    sar = s1.select(["VV", "VH"]).median()
    vv_db, vh_db = sar.select("VV"), sar.select("VH")
    ratio = vv_db.subtract(vh_db).unitScale(-5, 20).clamp(0, 1).rename("VVVH")
    vv = vv_db.unitScale(-25, 5).clamp(0, 1).rename("VV")
    vh = vh_db.unitScale(-30, 0).clamp(0, 1).rename("VH")

    grey = vv_db.unitScale(-25, 5).clamp(0, 1).multiply(31).toByte().rename("s")
    texture = grey.glcmTexture(size=3).select(["s_contrast", "s_var", "s_ent"])
    return composite.addBands([vv, vh, ratio]).addBands(texture)


def _build_collection(geometry, start_date, end_date, cloud_threshold):
    """Build a filtered & cloud-masked Sentinel-2 image collection for a geometry."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
    )
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


def build_sentinel_composite(geometry, start_date="2025-03-31", end_date="2025-04-30", cloud_threshold=15):
    """Build a Sentinel-2 median composite clipped to `geometry` with spectral indices."""
    init_ee()
    collection = _build_collection(geometry, start_date, end_date, cloud_threshold)
    composite = collection.median().clip(geometry)
    composite = _add_spectral_indices(composite)
    composite = _add_sar(composite, geometry, start_date, end_date)
    return composite


def sample_training_points(
    points,
    start_date="2025-03-31",
    end_date="2025-04-30",
    cloud_threshold=15,
):
    """Sample the canonical feature stack for a labelled point collection."""
    train_geometry = points.geometry().bounds()
    train_collection = _build_collection(
        train_geometry, start_date, end_date, cloud_threshold)
    train_composite = _add_spectral_indices(train_collection.median())
    train_composite = _add_sar(
        train_composite, train_geometry, start_date, end_date)
    return _sample_regions_with_geometry(train_composite.select(BANDS), points)


def _sample_regions_with_geometry(image, points):
    """Sample labelled points into an exportable table with point geometries."""
    return (
        image.sampleRegions(
            collection=points,
            properties=["label"],
            scale=10,
            tileScale=4,
            geometries=True,
        )
        .filter(ee.Filter.notNull(BANDS + ["label"]))
    )


def get_trained_classifier(model_type, start_date="2025-03-31", end_date="2025-04-30", cloud_threshold=15):
    """
    ALWAYS train on the composite covering the labelled training points.
    The classifier is then applied to any user-supplied AOI.
    """
    init_ee()

    cache_key = _classifier_cache_key(
        model_type, start_date, end_date, cloud_threshold)
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


def _static_overlay(classified, user_geometry, geometry_dict, max_px=2048):
    """Render the classified AOI once, as a single image.

    The XYZ endpoints above are *live inference*: every 256px tile the map asks
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
        url = classified.getThumbURL({
            "region": user_geometry,
            "dimensions": px,
            "crs": "EPSG:3857",
            "format": "png",
            "min": 0,
            "max": len(CLASS_PALETTE) - 1,
            "palette": CLASS_PALETTE,
        })
    except Exception as ex:
        print(f"Static overlay generation notice: {ex}")

    return {
        "url": url,
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "width_px": px,
    }


def classify_and_analyze(
    geometry_dict,
    model_type="rf",
    start_date="2025-03-31",
    end_date="2025-04-30",
    cloud_threshold=15,
    smoothing=True
):
    start_time = time.time()
    init_ee()

    # ── 1. Train on the labelled-point composite (training labels always live there) ──
    classifier = get_trained_classifier(model_type, start_date, end_date, cloud_threshold)

    # ── 2. Build the user's AOI composite for classification ──
    user_geometry = ee.Geometry(geometry_dict)
    aoi_composite = build_sentinel_composite(
        geometry=user_geometry,
        start_date=start_date,
        end_date=end_date,
        cloud_threshold=cloud_threshold
    )

    # ── 3. Classify the user's AOI ──
    classified = aoi_composite.select(BANDS).classify(classifier)
    if smoothing:
        classified = classified.focalMode(radius=1, kernelType="square", units="pixels")

    # ── 4. Generate GEE Tile URLs (streamed via XYZ tiles into Leaflet) ──
    rgb_map_id     = aoi_composite.getMapId({"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3})
    terrain_map_id = classified.getMapId(
        {"min": 0, "max": len(CLASS_PALETTE) - 1, "palette": CLASS_PALETTE})

    rgb_tile_url     = rgb_map_id["tile_fetcher"].url_format
    terrain_tile_url = terrain_map_id["tile_fetcher"].url_format

    overlay = _static_overlay(classified, user_geometry, geometry_dict)

    # ── 5. Compute individual class surface areas ──
    histogram = classified.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=user_geometry,
        scale=10,
        maxPixels=1e9
    ).getInfo()

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

    # ── 6. GeoTIFF export link ──
    download_url = ""
    try:
        download_url = classified.getDownloadURL({
            "name":   f"terrain_classified_{model_type}",
            "scale":  10,
            "crs":    "EPSG:4326",
            "region": user_geometry,
            "format": "GEO_TIFF"
        })
    except Exception as ex:
        print(f"GeoTIFF URL generation notice: {ex}")

    elapsed_sec = round(time.time() - start_time, 2)
    model_meta  = MODEL_METADATA.get(model_type.lower(), MODEL_METADATA["rf"])

    return {
        "status": "success",
        "model": {
            "id":                model_type,
            "name":              model_meta["name"],
            "type":              model_meta["type"],
            "description":       model_meta["description"],
            "internal_accuracy": model_meta["internal_accuracy"],
            "external_accuracy": model_meta["external_accuracy"]
        },
        "tile_urls": {
            "sentinel_rgb":       rgb_tile_url,
            "terrain_classified": terrain_tile_url
        },
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
