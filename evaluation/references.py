"""Independent whole-MP consensus reference construction.

WorldCover and Dynamic World must agree. Specialist datasets then confirm
water, buildings, and seasonal agriculture. Training points are used only to
buffer test samples away from them; evaluation is exclusively against these
public products.
"""

import ee


DW_MIN_PROBABILITY = 0.70
GHSL_MIN_BUILT_SQM = 50
# Same names as backend.config.LAND_COVER_CLASSES, so a confusion matrix built
# here and a class area reported by the dashboard use one vocabulary.
REFERENCE_LABELS = {
    0: "Vegetation",
    1: "Water",
    2: "Built Area",
    3: "Open Land",
    4: "Agriculture",
}
# Artefacts written before the rename spell three of the five classes
# differently. The integers never changed, so a stored result is still valid --
# it just needs its keys translated on the way in, which is cheaper and safer
# than re-running an hour of Earth Engine to relabel numbers that are correct.
LEGACY_REFERENCE_LABELS = {
    "Forest": "Vegetation",
    "Buildings": "Built Area",
    "Bare": "Open Land",
    "Barren Land": "Open Land",
    "Soil": "Open Land",
}


def canonical_label(name):
    """Current name for a class label, whatever spelling it arrives in."""
    return LEGACY_REFERENCE_LABELS.get(name, name)


# The two conditions do not share a label inventory, so each carries its own
# collapse onto the five-class reference taxonomy. Before is the preserved
# soil-era table: Soil (3) and Sand (4) both collapse to Bare, and it never
# learned Agriculture. After is the v3 inventory, already aligned with the
# reference except that Barren Land (3) reads as Bare.
PROJECT_TO_REFERENCE_LABEL = {
    "before": {0: 0, 1: 1, 2: 2, 3: 3, 4: 3},
    "after": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
}

WORLD_COVER_ID = "ESA/WorldCover/v200"
DYNAMIC_WORLD_ID = "GOOGLE/DYNAMICWORLD/V1"
WORLD_CEREAL_ID = "ESA/WorldCereal/2021/MODELS/v100"
WORLD_CEREAL_PRODUCT = "temporarycrops"
GHSL_ID = "JRC/GHSL/P2023A/GHS_BUILT_S_10m/2018"
OPERA_DSWX_ID = "OPERA/DSWX/L3_V1/HLS"
OPERA_WATER_BAND = "WTR_Water_classification"

DW_BANDS = (
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice",
)
WORLD_CEREAL_SEASON = {
    "winter": "tc-wintercereals",
    "summer": "tc-maize-main",
}


def collapse_project_prediction(values, condition):
    """Map one condition's project labels to the five-class reference taxonomy."""
    mapping = PROJECT_TO_REFERENCE_LABEL[condition]
    return [mapping[value] for value in values]


def _dynamic_world(region, start_date, end_date):
    probabilities = (
        ee.ImageCollection(DYNAMIC_WORLD_ID)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select(list(DW_BANDS))
        .mean()
    )
    label = probabilities.toArray().arrayArgmax().arrayGet([0]).rename("dw_label")
    probability = probabilities.reduce(ee.Reducer.max()).rename("dw_probability")
    return label.addBands(probability)


def _world_cereal(region, season):
    # The seasonal active-marker product has no valid crop pixels across MP for
    # the requested periods. Use WorldCereal's global temporary-crops product;
    # seasonality still comes from the date-matched Dynamic World/composite.
    return (
        ee.ImageCollection(WORLD_CEREAL_ID)
        .filterBounds(region)
        .filter(ee.Filter.eq("product", WORLD_CEREAL_PRODUCT))
        .select("classification")
        .mosaic()
        .eq(100)
    )


def _opera_water(region, start_date, end_date):
    water = (
        ee.ImageCollection(OPERA_DSWX_ID)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select(OPERA_WATER_BAND)
        .map(lambda image: image.updateMask(image.lt(252)))
        .mode()
    )
    return water.eq(1).Or(water.eq(2))


def build_consensus_reference(region, season, start_date, end_date):
    """Build a masked five-class, high-confidence consensus label image."""
    if season not in WORLD_CEREAL_SEASON:
        raise ValueError(f"Unknown season: {season}")

    world_cover = ee.ImageCollection(WORLD_COVER_ID).first().select("Map")
    dynamic_world = _dynamic_world(region, start_date, end_date)
    dw_label = dynamic_world.select("dw_label")
    confident = dynamic_world.select("dw_probability").gte(DW_MIN_PROBABILITY)
    ghsl = ee.Image(GHSL_ID).select("built_surface").gte(GHSL_MIN_BUILT_SQM)
    crops = _world_cereal(region, season)
    opera_water = _opera_water(region, start_date, end_date)

    specialist_conflict = opera_water.Or(ghsl).Or(crops)
    masks = (
        world_cover.eq(10).And(dw_label.eq(1)).And(confident),
        world_cover.eq(80).And(dw_label.eq(0)).And(confident).And(opera_water),
        world_cover.eq(50).And(dw_label.eq(6)).And(confident).And(ghsl),
        world_cover.eq(60).And(dw_label.eq(7))
        .And(specialist_conflict.Not()),
        world_cover.eq(40).And(crops),
    )
    reference = ee.Image(0).updateMask(masks[0]).rename("reference")
    for label, mask in enumerate(masks[1:], start=1):
        reference = reference.blend(ee.Image(label).updateMask(mask))
    return reference.toByte().clip(region)


def build_reference_image(region, season, start_date, end_date):
    """Public interface retained for runner and notebook use."""
    return build_consensus_reference(region, season, start_date, end_date)


def madhya_pradesh_districts():
    """Return GAUL level-2 districts inside Madhya Pradesh."""
    return (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", "Madhya Pradesh"))
    )


def exclude_training_neighbours(samples, training, distance_metres=100):
    """Remove candidates within ``distance_metres`` of any training point."""
    proximity = ee.Filter.withinDistance(
        distance=distance_metres,
        leftField=".geo",
        rightField=".geo",
        maxError=10,
    )
    return ee.FeatureCollection(
        ee.Join.inverted().apply(
            primary=ee.FeatureCollection(samples),
            secondary=ee.FeatureCollection(training),
            condition=proximity,
        )
    )


def sample_mp_reference(reference, season, training, points_per_class=20,
                        seed=20260807):
    """Sample each external class per MP district, then enforce leakage buffer."""
    districts = madhya_pradesh_districts()
    class_values = list(REFERENCE_LABELS)
    class_points = [points_per_class] * len(class_values)

    def add_district(district, accumulated):
        district = ee.Feature(district)
        sampled = reference.stratifiedSample(
            numPoints=0,
            classBand="reference",
            region=district.geometry(),
            scale=10,
            classValues=class_values,
            classPoints=class_points,
            seed=seed,
            geometries=True,
            tileScale=4,
        ).map(lambda feature: feature.set({
            "district": district.get("ADM2_NAME"),
            "season": season,
        }))
        return ee.FeatureCollection(accumulated).merge(sampled)

    samples = ee.FeatureCollection(
        districts.iterate(add_district, ee.FeatureCollection([]))
    )
    return exclude_training_neighbours(samples, training)
