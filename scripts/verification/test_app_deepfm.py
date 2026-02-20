import os
import sys
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import app

def test_deepfm_integration():
    print("Testing DeepFM integration...")
    
    # 1. Load Stacking Model
    try:
        app.load_model('stacking')
        print("Stacking model loaded successfully.")
    except Exception as e:
        print(f"Failed to load stacking model: {e}")
        return

    if 'stacking' not in app.MODELS:
        print("Error: Stacking model not found in app.MODELS.")
        return

    if app.DEEPFM_INFERENCE is None:
        print("Error: app.DEEPFM_INFERENCE not initialized.")
        return

    # 2. Create Dummy Data
    data = {
        '馬番': [1, 2],
        '枠番': [1, 2],
        '斤量': [55.0, 56.0],
        '体重': [480, 500],
        '体重変化': [0, -2],
        '性': ['牡', '牝'],
        '年齢': [3, 4],
        'horse_id': ['h1', 'h2'],
        'jockey_id': ['j1', 'j2'],
        'trainer_id': ['t1', 't2'],
        'sire': ['s1', 's2'],
        'dam': ['d1', 'd2'],
        'weather': ['晴', '曇'],
        'ground_state': ['良', '稍重'],
        'race_type': ['芝', 'ダート'],
        'venue_id': [1, 2],
        'course_len': [1600, 2000]
    }
    df = pd.DataFrame(data)
    
    # 3. Test DeepFM Inference directly
    print("Running DeepFM raw prediction...")
    try:
        scores = app.DEEPFM_INFERENCE.predict(df)
        print(f"DeepFM scores: {scores}")
        if len(scores) == 2:
            print("DeepFM raw prediction: SUCCESS")
        else:
            print(f"DeepFM raw prediction: FAILED (Expected 2 scores, got {len(scores)})")
    except Exception as e:
        print(f"DeepFM raw prediction error: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test app logic (mocking features if needed)
    print("Verifying feature names in stacking model...")
    model = app.MODELS['stacking']
    print(f"Model feature names: {model.feature_names}")
    if 'deepfm_score' in model.feature_names:
        print("Stacking model correctly requires 'deepfm_score'.")
    else:
        print("Warning: 'deepfm_score' not found in model features. Check if this is the correct model.")

if __name__ == "__main__":
    test_deepfm_integration()
