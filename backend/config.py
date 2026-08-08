import os
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT_ID = os.getenv("EE_PROJECT_ID")

DEFAULT_MAP_CENTER = {
    "lat": 20.5937,
    "lng": 78.9629,
    "zoom": 5,
    "label": "India"
}

JABALPUR_MAP_CENTER = {
    "lat": 23.1815,
    "lng": 79.9864,
    "zoom": 10,
    "label": "Jabalpur District Reference Area"
}

# Backwards compatibility alias
CAMPUS_MAP_CENTER = JABALPUR_MAP_CENTER

LAND_COVER_CLASSES = {
    0: {"name": "Forest", "color": "#006400", "description": "Dense trees, protected canopy & perennial foliage"},
    1: {"name": "Water", "color": "#0000FF", "description": "Rivers, reservoirs, lakes & permanent water bodies"},
    2: {"name": "Urban / Buildings", "color": "#FF0000", "description": "Concrete structures, built-up areas & roads"},
    3: {"name": "Barren Land / Exposed Earth", "color": "#B8860B", "description": "Rock outcrops, quarries, river sandbanks & permanently bare soil"},
    4: {"name": "Agricultural Land", "color": "#7CFC00", "description": "Cropland, tilled fields & seasonal farming plots"}
}

CLASS_PALETTE = ["006400", "0000FF", "FF0000", "B8860B", "7CFC00"]

DEFAULT_START_DATE = "2025-03-01"
DEFAULT_END_DATE = "2025-04-30"
DEFAULT_CLOUD_THRESHOLD = 15.0

BASE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI", "SAVI", "BSI"]

# Stacked 30-band feature space across three MP seasons
BANDS = (
    [f"{b}_winter" for b in BASE_BANDS] +
    [f"{b}_summer" for b in BASE_BANDS] +
    [f"{b}_postmonsoon" for b in BASE_BANDS]
)

FEATURE_COLLECTIONS = {
    "forest": "users/cosypix/jabalpur_forest_points",
    "water": "users/cosypix/jabalpur_water_points",
    "buildings": "users/cosypix/jabalpur_urban_points",
    "barren": "users/cosypix/jabalpur_barren_points",
    "agriculture": "users/cosypix/jabalpur_agriculture_points",
    "sand": "users/cosypix/jabalpur_sand_points"
}

# Regional Jabalpur boundary where training labels exist.
# The classifier is ALWAYS trained on this regional area, then applied to any user-drawn AOI.
JABALPUR_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [79.5000, 23.5500],
        [80.5000, 23.5500],
        [80.5000, 22.8000],
        [79.5000, 22.8000],
        [79.5000, 23.5500]
    ]]
}

# Backwards compatibility alias
CAMPUS_GEOJSON = JABALPUR_GEOJSON

MODEL_METADATA = {
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble Trees",
        "description": "100 decision trees trained with bagging fraction 0.3 & max nodes 75. High baseline accuracy.",
        "internal_accuracy": 99.35,
        "external_accuracy": 93.89,
        "params": {
            "numberOfTrees": 100,
            "minLeafPopulation": 1,
            "bagFraction": 0.3,
            "maxNodes": 75
        }
    },
    "svm": {
        "name": "Support Vector Machine (SVM)",
        "type": "Kernel-based Classifier",
        "description": "RBF kernel with C=100 & Gamma=1.0. Highest generalization accuracy on unseen terrain.",
        "internal_accuracy": 98.70,
        "external_accuracy": 98.43,
        "params": {
            "kernelType": "RBF",
            "gamma": 1.0,
            "cost": 100.0
        }
    },
    "xgb": {
        "name": "Gradient Boosted Trees (XGBoost)",
        "type": "Gradient Boosting",
        "description": "(Recommended) 100 gradient boosted trees with shrinkage rate 0.1. Strong generalization performance.",
        "internal_accuracy": 98.90,
        "external_accuracy": 97.27,
        "params": {
            "numberOfTrees": 100,
            "shrinkage": 0.1,
            "maxNodes": 10
        }
    },
    "cart": {
        "name": "Decision Tree (CART)",
        "type": "Single Tree",
        "description": "Classification and Regression Tree algorithm with maxNodes=20. Fast and interpretable baseline.",
        "internal_accuracy": 94.20,
        "external_accuracy": 94.14,
        "params": {
            "maxNodes": 20
        }
    },
    "knn": {
        "name": "K-Nearest Neighbors (KNN)",
        "type": "Instance-based",
        "description": "K=5 nearest neighbors using Euclidean distance across spectral index feature space.",
        "internal_accuracy": 93.50,
        "external_accuracy": 96.53,
        "params": {
            "k": 5
        }
    }
}
