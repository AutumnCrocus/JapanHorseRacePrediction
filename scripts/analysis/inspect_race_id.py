import pandas as pd
import pickle
import os

DATA_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'
RESULTS_FILE = os.path.join(DATA_DIR, 'results.pickle')

def inspect_format():
    with open(RESULTS_FILE, 'rb') as f:
        df = pickle.load(f)
    print("Column Types:")
    print(df.dtypes)
    if 'race_id' in df.columns:
        print("race_id head:")
        print(df['race_id'].head())
    else:
        print("race_id in index:")
        # Check specific 2025 IDs
        idx_str = df.index.astype(str)
        mask_2025 = idx_str.str.startswith('2025')
        print(f"2025 Count: {mask_2025.sum()}")
        if mask_2025.sum() > 0:
            ids = idx_str[mask_2025]
            rrs = ids.str[-2:].astype(int).unique()
            print(f"Unique Race Numbers (RR): {sorted(rrs)}")
        
if __name__ == '__main__':
    inspect_format()
