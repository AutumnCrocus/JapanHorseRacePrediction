import pickle
import pandas as pd
import os

RAW_DATA_DIR = 'data/raw'
HORSE_RESULTS_FILE = 'horse_results.pickle'

def inspect_hr():
    print("Loading horse_results.pickle...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    print(f"Loaded {len(hr)} entries.")
    
    keys = list(hr.keys())
    print(f"Keys (first 5): {keys[:5]}")
    
    for i in range(min(5, len(keys))):
        key = keys[i]
        val = hr[key]
        print(f"\nEntry for {key}: type={type(val)}")
        if isinstance(val, pd.DataFrame):
            print(f"Columns: {val.columns.tolist()}")
            print(val.head(2))
        elif isinstance(val, list):
            print(f"Sample item from list: {val[0] if val else 'empty'}")
        else:
            print(f"Value: {val}")

if __name__ == "__main__":
    inspect_hr()
