
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.betting_allocator import BettingAllocator

def test_allocator():
    # Create dummy predictions
    df_preds = pd.DataFrame({
        'horse_number': [1, 2, 3, 4, 5, 6],
        'probability': [0.4, 0.3, 0.1, 0.05, 0.05, 0.1],
        'odds': [None] * 6,
        'expected_value': [0] * 6,
        'horse_name': [f"H{i}" for i in range(1, 7)]
    })
    
    budget = 5000
    
    print("Calling allocate_budget...")
    try:
        recs = BettingAllocator.allocate_budget(df_preds, budget, odds_data=None)
        print(f"Recommendations count: {len(recs)}")
        for r in recs:
            print(r)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    test_allocator()
