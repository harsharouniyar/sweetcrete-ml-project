# SweetCrete ML Project

This repository contains a machine learning prototype for predicting concrete load strength in a PCC-focused low-carbon concrete context. The project is designed for clear, reproducible experimentation with a compact feature set and multiple regression baselines.

## Project Goal

Build and compare regression models to predict `Max_Load_lbf` from selected material and mix-related features, then save the best-performing model for reuse.

## Version 1 Feature Set

- `Log_Age`
- `WaterCement_Ratio`
- `PCC_Cement_Ratio`
- `Density_lb_per_in3`
- `Weight_lb`

## Version 1 Models

- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor (fallback: KNeighborsRegressor if XGBoost is unavailable)
- ElasticNet
- Linear Regression

## Folder Structure

- `data/` raw input dataset files
- `notebooks/` experimentation and training notebooks
- `src/` reusable Python source code (future expansion)
- `models/` saved trained model artifacts
- `outputs/` generated figures and evaluation outputs

## Run (Quick Start)

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Open and run `notebooks/model_training_v1.ipynb`.
4. Review `outputs/model_comparison_v1.png` and `models/best_model_v1.pkl`.
