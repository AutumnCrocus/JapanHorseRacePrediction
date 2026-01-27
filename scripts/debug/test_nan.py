import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from app import run_prediction_logic, load_model
import json

# Initialize model
load_model()

def test_nan_handling():
    print("Testing NaN handling in run_prediction_logic...")
    
    # Create a DataFrame with NaN in '単勝' and '人気'
    df = pd.DataFrame({
        '馬番': [1, 2],
        '馬名': ['Horse A', 'Horse B'],
        '単勝': [np.nan, 2.5],
        '人気': [np.nan, 1],
        '枠番': [1, 1],
        '斤量': [56.0, 56.0],
        '年齢': [4, 4],
        '体重': [480, 480],
        '体重変化': [0, 0],
        'course_len': [2000, 2000],
        'avg_rank': [5.0, 5.0],
        'win_rate': [0.1, 0.1],
        'place_rate': [0.3, 0.3],
        'race_count': [10, 10],
        'jockey_avg_rank': [5.0, 5.0],
        'jockey_win_rate': [0.1, 0.1],
        '性': [0, 0],
        'race_type': [0, 0]
    })
    
    try:
        response = run_prediction_logic(df, "Test Race", "Test Info")
        data = json.loads(response.data)
        print("Success! Response data:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Check if values are correctly handled
        for pred in data['predictions']:
            if pred['horse_name'] == 'Horse A':
                assert pred['odds'] == 0.0
                assert pred['popularity'] == 0
                print("Horse A (NaN input) handled correctly (odds=0.0, popularity=0)")
            if pred['horse_name'] == 'Horse B':
                assert pred['odds'] == 2.5
                assert pred['popularity'] == 1
                print("Horse B (Normal input) handled correctly")
                
    except Exception as e:
        print(f"Failed! Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nan_handling()
