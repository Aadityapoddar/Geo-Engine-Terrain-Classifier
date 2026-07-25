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
    3: {"name": "Soil", "color": "#D2B48C", "description": "Bare ground, open earth & unpaved terrain"}
}

CLASS_PALETTE = ["00FF00", "0000FF", "FF0000", "D2B48C"]

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI", "SAVI"]

FEATURE_COLLECTIONS = {
    "water": "users/cosypix/water_points",
    "forest": "users/cosypix/forest_points",
    "soil": "users/cosypix/soil_points",
    "buildings": "users/cosypix/building_points"
}

# The campus geometry where all training labels exist.
# The classifier is ALWAYS trained on this area, then applied to any user-drawn AOI.
CAMPUS_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [80.01710357666015, 23.173962177472703],
        [80.03259601593017, 23.165361215115187],
        [80.03654422760009, 23.172502420044232],
        [80.026802444458,   23.181694681000845],
        [80.01542987823485, 23.176960548201308],
        [80.01710357666015, 23.173962177472703]
    ]]
}

MODEL_METADATA = {
    "rf": {
        "name": "Random Forest",
        "type": "Ensemble Trees",
        "description": "200 decision trees trained with bagging fraction 0.3 & max nodes 10. Highest internal baseline accuracy.",
        "internal_accuracy": 99.35,
        "external_accuracy": 87.50,
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
        "description": "RBF kernel with C=100 & Gamma=1.0. Tied for highest generalization accuracy on unseen terrain.",
        "internal_accuracy": 98.70,
        "external_accuracy": 91.67,
        "params": {
            "kernelType": "RBF",
            "gamma": 1.0,
            "cost": 100.0
        }
    },
    "xgb": {
        "name": "Gradient Boosted Trees (XGBoost)",
        "type": "Gradient Boosting",
        "description": "100 gradient boosted trees with shrinkage rate 0.1. Strong generalization performance.",
        "internal_accuracy": 98.90,
        "external_accuracy": 91.67,
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
        "external_accuracy": 82.50,
        "params": {
            "maxNodes": 20
        }
    },
    "knn": {
        "name": "K-Nearest Neighbors (KNN)",
        "type": "Instance-based",
        "description": "K=5 nearest neighbors using Euclidean distance across spectral index feature space.",
        "internal_accuracy": 93.50,
        "external_accuracy": 79.17,
        "params": {
            "k": 5
        }
    }
}
