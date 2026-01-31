
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.betting_allocator import BettingAllocator

def test_fields():
    # Mock Recommendations (as output by allocate_budget before formatting)
    recommendations = [
        {
            'type': '3連単', 'amount': 2400, 'count': 24, 
            'horses': [1, 2, 3, 4], 'method': 'BOX', 'desc': 'BOX'
        }
    ]
    
    # Mock Predictions DataFrame
    data = {
        'horse_number': [1, 2, 3, 4],
        'horse_name': ['馬A', '馬B', '馬C', '馬D'], # Using 'horse_name' column
        'probability': [0.5, 0.4, 0.3, 0.2],
        'odds': [2.0, 3.0, 4.0, 5.0],
        'expected_value': [1.0, 1.2, 1.2, 1.0],
        '馬名': ['馬A', '馬B', '馬C', '馬D']  # Also checking fallback
    }
    df_preds = pd.DataFrame(data)
    
    # Run formatting
    final_list = BettingAllocator._format_recommendations(recommendations, df_preds)
    
    if not final_list:
        print("Error: No recommendations returned.")
        return

    rec = final_list[0]
    print("Formatted Recommendation:", rec)
    
    # Assertions
    assert 'horse_name' in rec, "Missing horse_name"
    assert rec['horse_name'] == '馬A', f"Incorrect horse_name: {rec.get('horse_name')}"
    
    assert 'odds' in rec, "Missing odds"
    assert rec['odds'] == 2.0, f"Incorrect odds: {rec.get('odds')}"
    
    assert 'prob' in rec, "Missing prob"
    assert rec['prob'] == 0.5, f"Incorrect prob: {rec.get('prob')}"
    
    assert 'ev' in rec, "Missing ev"
    assert rec['ev'] == 1.0, f"Incorrect ev: {rec.get('ev')}"
    
    print("SUCCESS: All fields are present and correct.")

if __name__ == "__main__":
    try:
        test_fields()
    except Exception as e:
        print(f"FAILED: {e}")
