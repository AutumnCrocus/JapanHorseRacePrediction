# Scripts Directory Structure

This directory contains various scripts for the JapanHorseRacePrediction project.

## 📂 core/
Production-ready scripts essential for the application lifecycle.
- `predict_tomorrow.py`: Main script for daily race prediction.
- `train_production.py`: Script to train the production model.
- `evaluate_prediction.py`: Tools for evaluating model performance.

## 📂 simulation/
Scripts for simulating betting strategies and calculating recovery rates.
- `simulate_graded_30patterns.py`: **Main script for Graded Race Strategy**
- `run_rolling_simulation.py`: Walk-forward validation script.

## 📂 analysis/
Tools for analyzing data distributions, prize money, and filtering logic.
- `analyze_graded_failure.py`: Analysis of model accuracy in graded races.

## 📂 debug/
Scripts for debugging, quick verification, and temporary checks.
- Contains one-off scripts and diagnostic tools.

## 📂 legacy/
Deprecated scripts kept for reference.

## Usage
When running scripts from subdirectories, ensure your python path is set correctly, although most scripts have been updated to handle import paths automatically.

Example:
```bash
python scripts/core/predict_tomorrow.py
```
