# Campus Land Cover Classification with Sentinel-2

This project performs binary land cover classification (Forest vs. Non-Forest) over a university campus area using Google Earth Engine and machine learning algorithms.

## Overview
Using Sentinel-2 Surface Reflectance data, the project filters images to a strict **1% cloud cover threshold (QA60 masking)** and evaluates three classifiers:
- **Random Forest**
- **Support Vector Machine (SVM) (RBF Kernel)**
- **XGBoost (Gradient Boosted Trees)**

## Key Findings
- Internal validation (Campus bounds) demonstrates **Random Forest** as the strongest baseline (99.35%).
- External validation (Test Area unseen land) demonstrates that **SVM** and **XGBoost** tie for the highest generalization accuracy (91.67%).
- **Winter imagery** provides the most accurate spectral separability for the study area.
- 250 points per label were used for internal training/validation (70/30 split), and 240 points per label were evaluated externally.

## Repository Structure
- `fe/`: Feature extraction and temporal/cloud sensitivity analysis notebooks.
- `nb/`:
  - `rf/`, `svm/`, `xgb/`: Classifier training, hyperparameter tuning, and spatial mapping notebooks.
  - `cal_area/`: Unseen test area validation notebooks for each classifier.
- `assets/`: Generated charts, heatmaps, and classified maps.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Google Earth Engine Authentication:**
   Authenticate your local Earth Engine environment:
   ```bash
   earthengine authenticate
   ```
   Ensure your `.env` file contains your Earth Engine Project ID:
   ```env
   EE_PROJECT_ID=your-project-id
   ```

3. **Run Notebooks:**
   Open Jupyter notebooks to explore feature extraction, hyperparameter grid search, and final classification outputs.
