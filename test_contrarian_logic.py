
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from modules.betting_allocator import BettingAllocator

def test_meta_contrarian():
    print("Testing meta_contrarian strategy...")
    
    # Create dummy data
    data = {
        'horse_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'probability': [0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01],
        'odds': [2.0, 4.0, 8.0, 10.0, 20.0, 25.0, 30.0, 50.0, 60.0, 100.0]
    }
    df_preds = pd.DataFrame(data)
    
    # Test case 1: High score (Solid)
    # Top prob = 0.4, odds = 2.0 -> Score should be high
    # top1_odds < 5 (+30)
    # top1_prob 0.4 (<0.7, +0)
    # prob_gap = 0.2 (+20)
    # Score = 50 -> A rank
    
    print("\nCase 1: Score ~50 (A rank)")
    recs = BettingAllocator._allocate_meta_contrarian(df_preds, 5000)
    for r in recs:
        print(r)
        
    # Test case 2: Low score (Chaotic)
    data2 = {
        'horse_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'probability': [0.15, 0.14, 0.13, 0.12, 0.10, 0.10, 0.08, 0.08, 0.05, 0.05],
        'odds': [6.0, 6.5, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0]
    }
    df_preds2 = pd.DataFrame(data2)
    
    # Top prob = 0.15, odds = 6.0
    # top1_odds < 10 (+20)
    # top1_prob < 0.7 (+0)
    # prob_gap = 0.01 (<0.1, +10)
    # Score = 30 -> B rank?
    
    print("\nCase 2: Score ~30 (B rank)")
    recs = BettingAllocator._allocate_meta_contrarian(df_preds2, 5000)
    for r in recs:
        print(r)
        
    # Test case 3: Very Low score (Super Chaotic)
    data3 = {
        'horse_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'probability': [0.11, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09],
        'odds': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    }
    df_preds3 = pd.DataFrame(data3)
    # top1_odds = 10.0 (+10)
    # top1_prob = 0.11 (+0)
    # prob_gap = 0.0 (+10)
    # Score = 20 -> B rank?
    
    # Let's try to make score < 20
    # Increase odds
    data4 = {
        'horse_number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'probability': [0.11, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09],
        'odds': [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    }
    df_preds4 = pd.DataFrame(data4)
    # top1_odds = 16.0 (+0)
    # prob_gap = 0 (+10)
    # Score = 10 -> C rank
    
    print("\nCase 4: Score 10 (C rank)")
    recs = BettingAllocator._allocate_meta_contrarian(df_preds4, 5000)
    for r in recs:
        print(r)

if __name__ == "__main__":
    test_meta_contrarian()
