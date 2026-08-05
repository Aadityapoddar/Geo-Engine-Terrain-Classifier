import os
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT_ID = os.getenv("EE_PROJECT_ID")
EE_ASSET_ROOT = os.getenv("EE_ASSET_ROOT", "users/ashutoshsaxena703")

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

# Spectral bands + indices, then GLCM texture.
# Texture is what separates bare soil from built-up: concrete is structurally rough
# (edges, roofs, shadows), bare soil is smooth. Brightness alone cannot tell them
# apart -- dry soil red reflectance 0.222 vs concrete 0.226.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI", "SAVI",
         "BSI", "UI", "IBI", "SWIRratio", "BAEI",
         "g_contrast", "g_ent", "g_var", "g_idm", "g_diss", "g_asm",
         "VV", "VH", "VVVH", "s_contrast", "s_var", "s_ent"]

# District-wide Jabalpur training points (1000 per class), not the old campus-only set.
FEATURE_COLLECTIONS = {
    "water": f"{EE_ASSET_ROOT}/jabalpur_water_points",
    "forest": f"{EE_ASSET_ROOT}/jabalpur_forest_points",
    "soil": f"{EE_ASSET_ROOT}/jabalpur_soil_points",
    "buildings": f"{EE_ASSET_ROOT}/jabalpur_building_points"
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
