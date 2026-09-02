import os
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT_ID = os.getenv("EE_PROJECT_ID")
if not EE_PROJECT_ID:
    raise RuntimeError(
        "EE_PROJECT_ID is not set. Copy .env.example to .env and fill it in."
    )

# Assets normally live under the project's own asset root, and that is the
# default. It is overridable because it has to be: this project's five class
# collections were imported before Earth Engine moved to project asset roots
# and still resolve only under the legacy `users/<account>` path. Deriving the
# root and ignoring EE_ASSET_ROOT made every collection unreachable --
# "Collection.loadTable: Collection asset ... not found" on the first classify
# -- while the /api/models and /api/health responses stayed green, because
# neither of those touches an asset.
EE_ASSET_ROOT = (
    os.getenv("EE_ASSET_ROOT") or f"projects/{EE_PROJECT_ID}/assets"
)


def _asset(key, default_name):
    """Path of one class's training asset.

    `EE_ASSET_<KEY>` in .env names the asset exactly, so renaming an asset in
    the project is a one-line edit. Without it the name is assumed unchanged
    under this project's asset root.
    """
    return os.getenv(f"EE_ASSET_{key.upper()}") or f"{EE_ASSET_ROOT}/{default_name}"

# Bump this whenever the class inventory or the band cut changes. It keys the
# trained-classifier cache and is stamped into every results artefact, so it is
# the only thing that makes a stale artefact detectable. v1 had Soil at label 3
# sourced from `jabalpur_soil_points` + `soil_points_mp`; v2 put Barren Land
# there; v3 drops Sand entirely, moving Agriculture from label 5 to label 4 so
# the inventory stays contiguous. Results from different versions are not
# comparable and must not be merged.
#
# v4 keeps v3's class inventory and band cut but changes the composite the bands
# are measured on, which invalidates every v3 number just as thoroughly: the two
# seasonal windows are now equal length, Sentinel-1 is restricted to a single
# orbit direction, and the experiment path no longer substitutes out-of-season
# imagery when a window is empty. It also renames the boosted-tree model from
# `xgb` to `gtb`, because ee.Classifier.smileGradientTreeBoost is Smile's
# gradient tree boosting and not the XGBoost of Chen and Guestrin.
TRAINING_SCHEMA_VERSION = "five-class-19-band-v5-drop-summer"

# Both windows are exactly 59 days. They were 59 and 31 before, which made the
# winter-to-summer accuracy drop uninterpretable: a shorter window sees fewer
# passes and fewer chances of a clear one, so a thinner composite is confounded
# with the season it is meant to isolate. Equal length is the cheapest way to
# take that explanation off the table; scripts/composite_depth_audit.py measures
# what is left, the per-district count of valid observations actually reaching
# each median.
#
# ee.Filter.date is half-open, so `end` is the first excluded day: winter covers
# 1 January - 28 February 2025 and summer 1 April - 29 May 2025. Summer used to
# start 31 March and stop 30 April; it now sits deeper in the pre-monsoon dry
# season, well clear of the rabi harvest at one end and of the mid-June monsoon
# onset at the other.
SEASONS = {
    "winter": {"start": "2025-01-01", "end": "2025-03-01"},
    "summer": {"start": "2025-04-01", "end": "2025-05-30"},
}

SEASON_WINDOW_DAYS = 59

# Best current whole-MP external model in both validated seasons. It also lowers
# Indore Water false positives versus RF on the same independent reference.
DEFAULT_MODEL = "gtb"

BEFORE_TRAINING_TABLE = _asset("before_table", "district_train_table")

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

# Class names are the ones the manuscript uses. The code used to say Forest,
# Buildings and Barren Land while the paper said Vegetation, Built area and
# Open land for the same integer labels, which meant a reader comparing the
# dashboard against Table V had to translate. Only the display names changed:
# the integers, the assets behind them and every measured result are the same.
# Earth Engine asset ids keep their historical spellings
# (`jabalpur_forest_points`, `jabalpur_barren_points`) because renaming a remote
# asset is not a display change -- it breaks any deployment reading the old id.
LAND_COVER_CLASSES = {
    0: {"name": "Vegetation", "color": "#006400", "description": "Trees, forest, plantations & other green cover"},
    1: {"name": "Water", "color": "#0000FF", "description": "Lakes, rivers, ponds & water bodies"},
    2: {"name": "Built Area", "color": "#FF0000", "description": "Built-up structures, concrete & urban infrastructure"},
    3: {"name": "Open Land", "color": "#D2B48C", "description": "Bare rock, exposed soil & unvegetated open ground"},
    4: {"name": "Agriculture", "color": "#90EE90", "description": "Cultivated fields, seasonal crops & fallow agricultural land"},
}

CLASS_PALETTE = [c["color"].lstrip("#") for c in LAND_COVER_CLASSES.values()]

# Spectral bands + indices, then GLCM texture.
# Texture is what separates bare soil from built-up: concrete is structurally rough
# (edges, roofs, shadows), bare soil is smooth. Brightness alone cannot tell them
# apart -- dry soil red reflectance 0.222 vs concrete 0.226.
BANDS = ["B4", "B8", "B11", "B12", "NDVI", "NDWI", "SAVI", "UI", "IBI",
         "SWIRratio", "g_contrast", "g_var", "g_idm", "VV", "VH", "VVVH",
         "s_contrast", "s_var", "s_ent"]

# The 27-band stack every feature decision started from, and the two
# intermediate cuts, kept here so the statewide evaluation can score all three
# through the same code path the shipped stack uses. BANDS above is CUT19, the
# stack that ships.
#
# Naming the alternatives in config rather than in the ablation script is what
# lets the feature-stack choice be made on the development districts and then
# confirmed once on the test districts (evaluation/splits.py). Previously the
# 27 -> 22 -> 19 reduction was chosen on the same statewide population it was
# then reported against, so the reported gain was partly the gain of having
# looked.
FULL_BANDS = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "NDVI", "NDWI", "NDBI", "SAVI",
    "BSI", "UI", "IBI", "SWIRratio", "BAEI",
    "g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm",
    "VV", "VH", "VVVH", "s_contrast", "s_var", "s_ent",
]

_CUT5 = ("B8", "NDVI", "NDWI", "NDBI", "s_ent")
_CUT8 = ("B2", "B4", "B8", "NDVI", "NDWI", "NDBI", "SAVI", "s_ent")

# The historical stacks, as fixed literals. "b19" used to be an alias for
# BANDS, which was true only while BANDS happened to be that cut: the moment a
# measured selection changed BANDS, the alias silently redefined a historical
# reference point and the assertion below caught it. A comparison table needs
# these to mean the same thing across every rerun, so they are pinned to the
# cuts that define them and BANDS is carried separately.
BAND_STACKS = {
    "b27": list(FULL_BANDS),
    "b22": [band for band in FULL_BANDS if band not in _CUT5],
    "b19": [band for band in FULL_BANDS if band not in _CUT8],
    "production": list(BANDS),
}

assert len(FULL_BANDS) == 27
assert len(BAND_STACKS["b22"]) == 22
assert len(BAND_STACKS["b19"]) == 19
assert set(BANDS) <= set(FULL_BANDS), sorted(set(BANDS) - set(FULL_BANDS))


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
    "forest": _asset("forest", "jabalpur_forest_points"),
    "water": _asset("water", "jabalpur_water_points"),
    "buildings": _asset("buildings", "jabalpur_building_points"),
    "barren": _asset("barren", "jabalpur_barren_points"),
    # 1,000 Agriculture points (label 4 since v3). `_labelled` because the
    # earlier `jabalpur_agriculture_points_updated` import carried no attributes
    # at all, so every feature had a null label, sampleRegions dropped all 1,000
    # rows, and the model trained without the class ever erroring.
    # Rebuild with scripts/export_agriculture_asset.py.
    "agriculture": _asset("agriculture", "jabalpur_agriculture_points_labelled"),
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

# After-condition results on the HELD-OUT TEST DISTRICTS only, which is a
# different and smaller population than the whole-state figure this table used
# to carry. Agreement with the five-class public-map consensus, not random-split
# training accuracy and not field accuracy.
#
# The test half excludes the four districts that contain training points and the
# fifteen development districts that the classifier choice was made on, so these
# are the only numbers here that were not selected on. tests/test_model_benchmarks.py
# pins them to doc/assets/mp_spatial_uncertainty_v6.json so config cannot drift
# from the run. Regenerate with scripts/spatial_uncertainty.py.
MODEL_BENCHMARKS = {
    "rf": {"winter": 86.44827586206897, "summer": 71.84249628528974},
    "svm": {"winter": 86.37931034482759, "summer": 68.61069836552748},
    "gtb": {"winter": 87.41379310344828, "summer": 73.84843982169392},
    "cart": {"winter": 80.55172413793103, "summer": 61.32986627043091},
    "knn": {"winter": 85.27586206896551, "summer": 69.65081723625556},
}


def _benchmark(model):
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "metric": "held_out_district_five_class_overall_accuracy",
        "scope": ("29 held-out MP test districts, FAO GAUL 2015 "
                  "level-2 vintage; public-map consensus, not field "
                  "ground truth"),
        **MODEL_BENCHMARKS[model],
    }

MODEL_METADATA = {
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble Trees",
        "description": '300 decision trees with bagging fraction 0.5 and no node cap.',
        "benchmark": _benchmark("rf"),
        "params": {
            'numberOfTrees': 300,
            'minLeafPopulation': 1,
            'bagFraction': 0.5,
            'maxNodes': None,
        }
    },
    "svm": {
        "name": "Support Vector Machine (SVM)",
        "type": "Kernel-based Classifier",
        "description": 'RBF kernel with C=10.0 and gamma=0.1.',
        "benchmark": _benchmark("svm"),
        "params": {
            'kernelType': 'RBF',
            'gamma': 0.1,
            'cost': 10.0,
        }
    },
    "gtb": {
        "name": "Smile Gradient Tree Boosting (GTB)",
        "type": "Gradient Boosting",
        "description": (
            "100 gradient-boosted trees with shrinkage rate 0.1, via "
            "ee.Classifier.smileGradientTreeBoost."
        ),
        "benchmark": _benchmark("gtb"),
        "params": {
            'numberOfTrees': 300,
            'shrinkage': 0.1,
            'maxNodes': 10,
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
        "description": 'K=9 nearest neighbours over the feature space.',
        "benchmark": _benchmark("knn"),
        "params": {
            'k': 9,
        }
    }
}
