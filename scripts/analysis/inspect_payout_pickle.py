
import pickle
import os

path = "data/raw/payouts_verify_20260208.pkl"
if os.path.exists(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
        print(f"Loaded {len(data)} races.")
        if data:
            first_key = list(data.keys())[0]
            print(f"Sample race {first_key}: {data[first_key]}")
else:
    print("File not found.")
