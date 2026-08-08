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
        JABALPUR_GEOJSON,
        CAMPUS_GEOJSON,
        MODEL_METADATA,
        BASE_BANDS,
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
        JABALPUR_GEOJSON,
        CAMPUS_GEOJSON,
        MODEL_METADATA,
        BASE_BANDS,
    )

_EE_INITIALIZED = False


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
    """Add NDVI, NDWI, NDBI, SAVI index bands to a composite."""
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    savi = composite.expression(
        "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        {"NIR": composite.select("B8"), "RED": composite.select("B4")}
    ).rename("SAVI")
    
    bsi = composite.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {
            "SWIR1": composite.select("B11"),
            "RED": composite.select("B4"),
            "NIR": composite.select("B8"),
            "BLUE": composite.select("B2")
        }
    ).rename("BSI")
    
    return composite.addBands([ndvi, ndwi, ndbi, savi, bsi])


def _build_collection(geometry, start_date, end_date, cloud_threshold):
    """Build a filtered & cloud-masked Sentinel-2 image collection for a geometry."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(mask_s2_clouds)
    )
    # Fallback to a wider window if the strict filter yields no images
    count = collection.size().getInfo()
    if count == 0:
        print(f"No images in [{start_date} – {end_date}] with cloud < {cloud_threshold}%. Using fallback date range.")
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate("2024-01-01", "2025-12-31")
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(mask_s2_clouds)
        )
    return collection


def _add_topography(composite, geometry):
    """
    Adds SRTM elevation and slope bands to help the classifier
    distinguish low-lying river beds from elevated urban structures.

    NOTE: The DEM must NOT be clipped before computing slope.
    ee.Terrain.slope() requires a full 3x3 pixel neighbourhood.
    Clipping first masks out boundary neighbours, producing corrupted
    slope values across the entire (small) user AOI on the live website.
    Adding un-clipped bands to an already-clipped composite automatically
    bounds them to the composite's footprint.
    """
    dem = ee.Image("USGS/SRTMGL1_003")   # global DEM — do NOT clip here
    elevation = dem.select("elevation").rename("ELEVATION")
    slope = ee.Terrain.slope(dem).rename("SLOPE")   # neighbours intact
    return composite.addBands([elevation, slope])


def build_sentinel_composite(geometry, start_date="2025-03-01", end_date="2025-04-30", cloud_threshold=15):
    """Build a Sentinel-2 median composite clipped to `geometry` with spectral indices."""
    init_ee()
    collection = _build_collection(geometry, start_date, end_date, cloud_threshold)
    composite = collection.median().clip(geometry)
    composite = _add_spectral_indices(composite)
    composite = _add_topography(composite, geometry)
    return composite


def build_multi_temporal_stack(geometry, year="2025", cloud_threshold=50):
    """
    Builds a 30-band Multi-Temporal Stack combining Winter, Summer, 
    and Post-Monsoon Sentinel-2 composites with spectral indices.
    """
    seasons = {
        "winter": (f"{int(year)-1}-12-01", f"{year}-02-28"),
        "summer": (f"{year}-03-01", f"{year}-05-31"),
        "postmonsoon": (f"{year}-09-01", f"{year}-11-30")
    }
    
    stacked_image = None
    
    for season_name, (start_date, end_date) in seasons.items():
        collection = _build_collection(geometry, start_date, end_date, cloud_threshold)
        composite = collection.median().clip(geometry)
        composite = _add_spectral_indices(composite)
        composite = _add_topography(composite, geometry)
        
        seasonal_composite = composite.select(BASE_BANDS)
        new_names = [f"{band}_{season_name}" for band in BASE_BANDS]
        seasonal_composite = seasonal_composite.rename(new_names)
        
        if stacked_image is None:
            stacked_image = seasonal_composite
        else:
            stacked_image = stacked_image.addBands(seasonal_composite)
            
    return stacked_image


def get_trained_classifier(model_type, start_date="2025-03-01", end_date="2025-04-30", cloud_threshold=15):
    """
    ALWAYS train on the Jabalpur regional composite where labelled training points exist.
    The classifier is then applied to any user-supplied AOI.
    """
    init_ee()

    # ── Training geometry: Jabalpur region ──
    training_geometry = ee.Geometry(JABALPUR_GEOJSON)
    year = start_date[:4] if start_date else "2025"
    training_composite = build_multi_temporal_stack(
        geometry=training_geometry,
        year=year,
        cloud_threshold=50.0
    )

    # ── Load training label feature collections & set numeric class IDs explicitly ──
    forest      = ee.FeatureCollection(FEATURE_COLLECTIONS["forest"]).map(lambda f: f.set("label", 0))
    water       = ee.FeatureCollection(FEATURE_COLLECTIONS["water"]).map(lambda f: f.set("label", 1))
    buildings   = ee.FeatureCollection(FEATURE_COLLECTIONS["buildings"]).map(lambda f: f.set("label", 2))
    
    # Merge sand points into barren land
    barren_base = ee.FeatureCollection(FEATURE_COLLECTIONS["barren"])
    sand_base   = ee.FeatureCollection(FEATURE_COLLECTIONS["sand"])
    barren      = barren_base.merge(sand_base).map(lambda f: f.set("label", 3))
    
    agriculture = ee.FeatureCollection(FEATURE_COLLECTIONS["agriculture"]).map(lambda f: f.set("label", 4))

    all_points = forest.merge(water).merge(buildings).merge(barren).merge(agriculture)

    # ── Instantiate the requested classifier ──
    m = model_type.lower()
    
    training_bands = BANDS.copy()
    if m in ["svm", "knn"]:
        training_bands = [b for b in training_bands if "ELEVATION" not in b and "SLOPE" not in b]

    # ── Sample spectral values at training point locations ──
    training_samples = (
        training_composite.select(training_bands)
        .sampleRegions(
            collection=all_points,
            properties=["label"],
            scale=10,
            tileScale=4
        )
        .filter(ee.Filter.notNull(training_bands + ["label"]))
    )

    if m == "rf":
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=100, minLeafPopulation=1, bagFraction=0.3, maxNodes=75
        )
    elif m == "svm":
        classifier = ee.Classifier.libsvm(kernelType="RBF", gamma=1.0, cost=100.0)
    elif m == "xgb":
        classifier = ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=100, shrinkage=0.1, maxNodes=10
        )
    elif m == "cart":
        classifier = ee.Classifier.smileCart(maxNodes=20)
    elif m == "knn":
        classifier = ee.Classifier.smileKNN(k=5)
    else:
        classifier = ee.Classifier.smileRandomForest(numberOfTrees=100, maxNodes=75)

    trained_classifier = classifier.train(
        features=training_samples,
        classProperty="label",
        inputProperties=training_bands
    )
    return trained_classifier, training_bands


def classify_and_analyze(
    geometry_dict,
    model_type="rf",
    start_date="2025-03-01",
    end_date="2025-04-30",
    cloud_threshold=15,
    smoothing=True
):
    start_time = time.time()
    init_ee()

    # ── 1. Train on Jabalpur regional composite (training labels live there) ──
    classifier, training_bands = get_trained_classifier(model_type, start_date, end_date, cloud_threshold)

    # ── 2. Build the user's AOI composite for classification ──
    user_geometry = ee.Geometry(geometry_dict)
    year = start_date[:4] if start_date else "2025"
    aoi_composite = build_multi_temporal_stack(
        geometry=user_geometry,
        year=year,
        cloud_threshold=cloud_threshold
    )

    # ── 3. Classify the user's AOI ──
    classified = aoi_composite.select(training_bands).classify(classifier)
    if smoothing:
        classified = classified.focalMode(radius=1, kernelType="square", units="pixels")

    # ── 4. Generate GEE Tile URLs (streamed via XYZ tiles into Leaflet) ──
    rgb_map_id     = aoi_composite.getMapId({"bands": ["B4_winter", "B3_winter", "B2_winter"], "min": 0, "max": 0.3})
    terrain_map_id = classified.getMapId({"min": 0, "max": 4, "palette": CLASS_PALETTE})

    rgb_tile_url     = rgb_map_id["tile_fetcher"].url_format
    terrain_tile_url = terrain_map_id["tile_fetcher"].url_format

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
    for class_id in range(5):
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
