
import pickle
import os

PAYOUTS_PATH = 'data/raw/payouts_2025.pkl'
TARGET_ID = '202501010101'

if os.path.exists(PAYOUTS_PATH):
    with open(PAYOUTS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    if TARGET_ID in data:
        print(f"--- Data for {TARGET_ID} ---")
        payouts = data[TARGET_ID]
        print(f"Keys: {list(payouts.keys())}")
        
        for k, v in payouts.items():
            print(f"\nType: {k}")
            print(f"Content: {v}")
            # Check key type
            if v:
                sample_key = list(v.keys())[0]
                print(f"Sample Key: {repr(sample_key)} Type: {type(sample_key)}")
                if isinstance(sample_key, tuple):
                    print(f"Sample Key Element Type: {type(sample_key[0])}")
    else:
        print(f"Target {TARGET_ID} not found in payouts.")
        print(f"Sample keys: {list(data.keys())[:5]}")
else:
    print("Payout file not found.")
