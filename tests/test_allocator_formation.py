import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.betting_allocator import BettingAllocator

def create_mock_preds(distribution='normal', top_odds=3.0):
    if distribution == 'flat':
        # Flat distribution (confusion)
        probs = [0.18, 0.17, 0.16, 0.15, 0.14, 0.10, 0.05, 0.05]
    elif distribution == 'strong':
        # Strong axis
        probs = [0.45, 0.20, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
    else:
        # Normal
        probs = [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]
        
    data = []
    for i, p in enumerate(probs):
        data.append({
            'horse_number': i+1,
            'horse_name': f'Horse{i+1}',
            'probability': p,
            'odds': top_odds if i==0 else 10.0 + i
        })
    return pd.DataFrame(data)

def test_formation():
    print("=== Testing Smart Formation Allocator ===")
    
    # 1. Low Budget Test (500 yen)
    print("\n--- Case 1: Low Budget (500 yen) ---")
    df = create_mock_preds('normal')
    recs = BettingAllocator.allocate_budget(df, 500, strategy='formation')
    print(f"Result Count: {len(recs)}")
    for r in recs:
        print(f"Type: {r['bet_type']}, Method: {r['method']}, Cost: {r['total_amount']}")
    
    if not recs:
        print("FAIL: No bets for low budget")
    else:
        print("PASS: Fallback successful")

    # 2. Strong Axis & High Budget (10000 yen) -> Expect 3-Ren-Tan + Insurance
    print("\n--- Case 2: Strong Axis + High Budget (10000 yen) ---")
    df = create_mock_preds('strong', top_odds=25.0)  # EV = 0.45 * 25 = 11.25 >= 10
    recs = BettingAllocator.allocate_budget(df, 10000, strategy='formation')
    
    types = [r['bet_type'] for r in recs]
    print(f"Types: {types}")
    if '3連単' in types and '単勝' in types:
         print("PASS: 3-Ren-Tan and Insurance generated")
    else:
         print("FAIL: Expected 3-Ren-Tan and Insurance")
         
    # 3. Flat Distribution (Confused) -> Expect BOX
    print("\n--- Case 3: Flat Distribution ---")
    df = create_mock_preds('flat')
    recs = BettingAllocator.allocate_budget(df, 5000, strategy='formation')
    
    methods = [r['method'] for r in recs]
    print(f"Methods: {methods}")
    if any('BOX' in m for m in methods):
        print("PASS: BOX strategy selected")
    else:
        print("FAIL: Expected BOX")

if __name__ == "__main__":
    test_formation()
