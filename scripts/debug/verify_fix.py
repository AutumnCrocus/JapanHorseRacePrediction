import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from modules.data_loader import fetch_and_process_race_data
import os
import pickle
from modules.constants import MODEL_DIR
from modules.training import HorseRaceModel

def test_prediction():
    race_id = "202608010511"
    try:
        print(f"Testing prediction for race {race_id}...")
        df = fetch_and_process_race_data(race_id)
        
        # Check dtypes
        print("\nProcessed DataFrame Dtypes:")
        print(df.dtypes[['枠番', '馬番', '性']])
        
        # Load model to get feature names
        model_path = os.path.join(MODEL_DIR, 'horse_race_model.pkl')
        if os.path.exists(model_path):
            model = HorseRaceModel()
            model.load(model_path)
            feature_names = model.feature_names
            
            # Prepare X
            X = df[feature_names].copy()
            X = X.fillna(0)
            
            print("\nFeature Matrix X Dtypes for problematic columns:")
            print(X.dtypes[['枠番', '馬番', '性']])
            
            # Try predict
            print("\nAttempting prediction...")
            probs = model.predict(X)
            print(f"Prediction success! Probs: {probs[:3]}")
        else:
            print(f"Model not found at {model_path}, skipping prediction test but dtypes are verified.")
            
    except Exception as e:
        print(f"\nPrediction failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
