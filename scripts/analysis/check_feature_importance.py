
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.training import HorseRaceModel
from modules.constants import MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2026.pkl')

def analyze_importance():
    print("=== Feature Importance Analysis ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    try:
        model = HorseRaceModel()
        model.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Model Type: {model.model_type}")
    print(f"Total features in model: {len(model.feature_names)}")
    if 'deepfm_score' in model.feature_names:
        print("SUCCESS: 'deepfm_score' exists in model features.")
    else:
        print("FAILURE: 'deepfm_score' is MISSING from model features.")
        # Print first few features
        print("Sample features:", model.feature_names[:10])
    
    # Get Importance
    try:
        imp_df = model.get_feature_importance(top_n=50)
        print("\nTop 20 Features:")
        print(imp_df.head(20))
        
        # Check specific features
        search_feats = ['deepfm_score']
        print("\nTarget Features Rank:")
        for feat in search_feats:
            if feat in imp_df['feature'].values:
                rank = imp_df[imp_df['feature'] == feat].index[0] + 1
                score = imp_df[imp_df['feature'] == feat]['importance'].values[0]
                print(f"- {feat}: Rank {rank} (Score: {score:.4f})")
            else:
                print(f"- {feat}: Not found in top 50 (or not used)")
                
        # Save to CSV
        output_csv = os.path.join(os.path.dirname(__file__), '../../reports/feature_importance_2026.csv')
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        imp_df.to_csv(output_csv, index=False)
        print(f"\nSaved full importance list to {output_csv}")

    except Exception as e:
        print(f"Error analyzing importance: {e}")

if __name__ == "__main__":
    analyze_importance()
