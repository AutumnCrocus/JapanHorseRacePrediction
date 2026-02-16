
import pandas as pd
import pickle
import os

PAYOUTS_PATH = 'data/raw/payouts_2025.pkl'
PRED_PATH = 'data/processed/prediction_2025_ltr.csv'
TARGET_ID = '202505010101'

print(f"--- Inspecting {TARGET_ID} ---")

# 1. Payouts
if os.path.exists(PAYOUTS_PATH):
    with open(PAYOUTS_PATH, 'rb') as f:
        payouts = pickle.load(f)
    
    if TARGET_ID in payouts:
        print(f"Payout Found.")
        print(f"Tan: {payouts[TARGET_ID].get('tan')}")
        print(f"Sanrenpuku: {payouts[TARGET_ID].get('sanrenpuku')}")
        print(f"Tan Keys Type: {[type(k) for k in payouts[TARGET_ID].get('tan', {}).keys()]}")
    else:
        print(f"Payout NOT Found for {TARGET_ID}.")
        
        all_keys = sorted(list(payouts.keys()))
        print(f"Total Keys: {len(all_keys)}")
        print(f"First 10 Keys: {all_keys[:10]}")
        print(f"Last 10 Keys: {all_keys[-10:]}")
        
        # Check for any key containing '202505'
        contains_tokyo = [k for k in all_keys if '202505' in k]
        print(f"Keys containing '202505': {len(contains_tokyo)}")
        if contains_tokyo:
            print(f"Sample: {contains_tokyo[:5]}")
else:
    print("Payout file not found.")

# 2. Prediction
if os.path.exists(PRED_PATH):
    df = pd.read_csv(PRED_PATH)
    # Cast
    if 'race_id' in df.columns:
        df['race_id'] = df['race_id'].astype(str)
    
    subset = df[df['race_id'] == TARGET_ID]
    if not subset.empty:
        print(f"Prediction Found. Rows: {len(subset)}")
        print(subset[['horse_number', 'ltr_score', 'actual_rank']].sort_values('ltr_score', ascending=False).head(5))
        
        # Check horse_number type
        h_num = subset['horse_number'].iloc[0]
        print(f"Horse Number Sample: {h_num} Type: {type(h_num)}")
    else:
        print(f"Prediction NOT Found.")
else:
    print("Prediction file not found.")
