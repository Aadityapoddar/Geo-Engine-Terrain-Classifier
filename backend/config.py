import os
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT_ID = os.getenv("EE_PROJECT_ID")
EE_ASSET_ROOT = os.getenv("EE_ASSET_ROOT", "users/ashutoshsaxena703")

# Bump this whenever the class inventory or the band cut changes. It keys the
# trained-classifier cache and is stamped into every results artefact, so it is
# the only thing that makes a stale artefact detectable. v1 had Soil at label 3
# sourced from `jabalpur_soil_points` + `soil_points_mp`; v2 put Barren Land
# there; v3 drops Sand entirely, moving Agriculture from label 5 to label 4 so
# the inventory stays contiguous. Results from different versions are not
# comparable and must not be merged.
TRAINING_SCHEMA_VERSION = "five-class-19-band-v3-no-sand"

SEASONS = {
    "winter": {"start": "2025-01-01", "end": "2025-02-28"},
    "summer": {"start": "2025-03-31", "end": "2025-04-30"},
}

# Best current whole-MP external model in both validated seasons. It also lowers
# Indore Water false positives versus RF on the same independent reference.
DEFAULT_MODEL = "xgb"

BEFORE_TRAINING_TABLE = f"{EE_ASSET_ROOT}/district_train_table"

DEFAULT_MAP_CENTER = {
    "lat": 20.5937,
    "lng": 78.9629,
    "zoom": 5,
    "label": "India"
}

CAMPUS_MAP_CENTER = {
    "lat": 23.174,
    "lng": 80.026,
    "zoom": 15,
    "label": "Jabalpur Campus Study Area"
}

LAND_COVER_CLASSES = {
    0: {"name": "Forest", "color": "#00FF00", "description": "Trees, dense vegetation & foliage"},
    1: {"name": "Water", "color": "#0000FF", "description": "Lakes, rivers, ponds & water bodies"},
    2: {"name": "Buildings", "color": "#FF0000", "description": "Built-up structures, concrete & urban infrastructure"},
    3: {"name": "Barren Land", "color": "#D2B48C", "description": "Bare rock, exposed earth & unvegetated open ground"},
    4: {"name": "Agriculture", "color": "#F59E0B", "description": "Cultivated fields, seasonal crops & fallow agricultural land"},
}

CLASS_PALETTE = [c["color"].lstrip("#") for c in LAND_COVER_CLASSES.values()]

# Spectral bands + indices, then GLCM texture.
# Texture is what separates bare soil from built-up: concrete is structurally rough
# (edges, roofs, shadows), bare soil is smooth. Brightness alone cannot tell them
# apart -- dry soil red reflectance 0.222 vs concrete 0.226.
BANDS = ["B3", "B11", "B12", "BSI", "UI", "IBI", "SWIRratio", "BAEI",
         "g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm",
         "VV", "VH", "VVVH", "s_contrast", "s_var"]

# District-wide Jabalpur training points, not the old campus-only set, plus
# Agriculture.
#
# Class 3 is Barren Land, sourced from `jabalpur_barren_points`. It replaces
# the previous pair of Soil assets, `jabalpur_soil_points` and `soil_points_mp`.
# Be aware of what that trades away: `soil_points_mp` was 1,000 statewide points
# and the measured reason Soil recall across Madhya Pradesh went from 7% to 54%,
# because Jabalpur labels alone do not cover the bright dry bare ground of Malwa,
# Chambal and Bundelkhand -- the terrain the model reads as built-up. Barren Land
# is Jabalpur-only, so statewide bare-ground recall is expected to regress.
#
# Sand (`sand_points_mp_labelled`, previously label 4) was dropped in v3: no
# public reference product distinguishes river sand from bare ground, so the
# class could never be independently tested, and it competed with Barren Land
# for the same spectral space.
FEATURE_COLLECTIONS = {
    "forest": f"{EE_ASSET_ROOT}/jabalpur_forest_points",
    "water": f"{EE_ASSET_ROOT}/jabalpur_water_points",
    "buildings": f"{EE_ASSET_ROOT}/jabalpur_building_points",
    "barren": f"{EE_ASSET_ROOT}/jabalpur_barren_points",
    # 1,000 Agriculture points (label 4 since v3). `_labelled` because the
    # earlier `jabalpur_agriculture_points_updated` import carried no attributes
    # at all, so every feature had a null label, sampleRegions dropped all 1,000
    # rows, and the model trained without the class ever erroring.
    # Rebuild with scripts/export_agriculture_asset.py.
    "agriculture": f"{EE_ASSET_ROOT}/jabalpur_agriculture_points_labelled",
}

EXPECTED_ASSET_LABELS = {
    "forest": 0,
    "water": 1,
    "buildings": 2,
    "barren": 3,
    "agriculture": 4,
}

EXPECTED_ASSET_COUNTS = {
    "barren": 1000,
    "agriculture": 1000,
}

# Current v3 After-condition whole-Madhya-Pradesh results. These are agreement
# with the five-class public-map consensus, not random-split training accuracy.
# Source: doc/assets/mp_external_results_v3.json (96/96 district-season shards).
MODEL_BENCHMARKS = {
    "rf": {"winter": 78.04470440777105, "summer": 62.02723146747352},
    "svm": {"winter": 71.33904324211406, "summer": 61.14112816079533},
    "xgb": {"winter": 84.98015458533528, "summer": 70.45601901880268},
    "cart": {"winter": 71.35993315228745, "summer": 61.14112816079533},
    "knn": {"winter": 73.67871318153331, "summer": 61.33563864274908},
}


def _benchmark(model):
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "metric": "whole_mp_external_five_class_overall_accuracy",
        "scope": "48 MP districts; public-map consensus; not field ground truth",
        **MODEL_BENCHMARKS[model],
    }

MODEL_METADATA = {
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble Trees",
        "description": "200 decision trees with bagging fraction 0.3 and max nodes 10.",
        "benchmark": _benchmark("rf"),
        "params": {
            "numberOfTrees": 200,
            "minLeafPopulation": 1,
            "bagFraction": 0.3,
            "maxNodes": 10
        }
    },
    "svm": {
        "name": "Support Vector Machine (SVM)",
        "type": "Kernel-based Classifier",
        "description": "RBF kernel with C=100 and gamma=1.0.",
        "benchmark": _benchmark("svm"),
        "params": {
            "kernelType": "RBF",
            "gamma": 1.0,
            "cost": 100.0
        }
    },
    "xgb": {
        "name": "Gradient Boosted Trees (XGBoost)",
        "type": "Gradient Boosting",
        "description": "100 gradient-boosted trees with shrinkage rate 0.1.",
        "benchmark": _benchmark("xgb"),
        "params": {
            "numberOfTrees": 100,
            "shrinkage": 0.1,
            "maxNodes": 10
        }
    },
    "cart": {
        "name": "Decision Tree (CART)",
        "type": "Single Tree",
        "description": "Classification and Regression Tree with max nodes 20.",
        "benchmark": _benchmark("cart"),
        "params": {
            "maxNodes": 20
        }
    },
    "knn": {
        "name": "K-Nearest Neighbors (KNN)",
        "type": "Instance-based",
        "description": "K=5 nearest neighbors over the canonical 19-band feature space.",
        "benchmark": _benchmark("knn"),
        "params": {
            "k": 5
        }
    }
}
