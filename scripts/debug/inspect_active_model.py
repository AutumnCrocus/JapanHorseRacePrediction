
import sys
import os
import pickle
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from modules.constants import MODEL_DIR
from modules.training import HorseRaceModel, EnsembleModel

def get_active_model_path():
    production_model_path = os.path.join(MODEL_DIR, 'production_model.pkl')
    if os.path.exists(production_model_path):
        return production_model_path
    
    ensemble_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pkl')]
    if len(ensemble_files) > 1:
        return "ENSEMBLE_DIR"
        
    return os.path.join(MODEL_DIR, 'horse_race_model.pkl')

def inspect_model():
    path = get_active_model_path()
    
    model = None
    if path == "ENSEMBLE_DIR":
        model = EnsembleModel()
        model.load(MODEL_DIR)
    elif os.path.exists(path):
        model = HorseRaceModel()
        model.load(path)
    else:
        print("No model found!")
        return

    # 1. Algorithm
    algo = "Unknown"
    if isinstance(model, EnsembleModel):
        algo = "Ensemble (LGBM + RF)"
    else:
        algo = getattr(model, 'model_type', 'lgbm') # default to lgbm if attr missing
        # Map nice names
        algo_map = {
            'lgbm': 'LightGBM',
            'rf': 'Random Forest', 
            'gbc': 'Gradient Boosting',
            'xgb': 'XGBoost',
            'catboost': 'CatBoost',
            'pytorch_mlp': 'PyTorch Neural Network'
        }
        algo = algo_map.get(algo, algo)
    print(f"Algorithm: {algo}")

    # 2. Last Updated
    last_updated = "-"
    if path != "ENSEMBLE_DIR" and os.path.exists(path):
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        last_updated = dt.strftime('%Y/%m/%d %H:%M')
    print(f"Last Updated: {last_updated}")

    # 3. Feature Count
    feat_count = 0
    if model.feature_names:
        feat_count = len(model.feature_names)
    print(f"Feature Count: {feat_count}")
    
    # Also print top 5 features just in case
    print("\nTop 5 Features:")
    try:
        if isinstance(model, EnsembleModel):
            imp = model.get_feature_importance(5)
        else:
            imp = model.get_feature_importance(5)
        print(imp)
    except:
        print("Could not get features")

if __name__ == "__main__":
    inspect_model()
