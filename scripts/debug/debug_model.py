"""
Debug script to inspect saved model file
"""
import sys
import os
import pickle
import pandas as pd

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.training import HorseRaceModel
from modules.constants import MODEL_DIR

def inspect_model():
    model_path = os.path.join(MODEL_DIR, 'production_model.pkl')
    print(f"Checking model at: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    try:
        model = HorseRaceModel()
        model.load(model_path)
        
        print(f"Model Type: {model.model_type}")
        print(f"Feature Names: {len(model.feature_names) if model.feature_names else 'None'}")
        
        if model.feature_names:
            print(f"Top 5 Features: {model.feature_names[:5]}")
            
        print(f"Feature Importance Data:")
        if model.feature_importance is not None:
            print(model.feature_importance.head())
            print(model.feature_importance.dtypes)  # Check types
            print(f"Total Importance: {model.feature_importance['importance'].sum()}")
        else:
            print("Feature importance is None")
            
        # Re-generate importance if missing
        if model.feature_importance is None or model.feature_importance['importance'].sum() == 0:
            print("Attempting to regenerate feature importance...")
            if model.model:
                model._set_feature_importance()
                print(" Regenerated:")
                print(model.feature_importance.head())
                
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()
