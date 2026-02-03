
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, HORSE_RESULTS_FILE

file_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
with open(file_path, 'rb') as f:
    df = pickle.load(f)

df.columns = df.columns.str.replace(' ', '')
if 'レース名' in df.columns:
    print("Searching for Nakayama 2025 examples in 'レース名'...")
    # Filter for Nakayama and 2025
    # Since we suspect format is "YY/MM/DD Place ..."
    # Look for "25/01" and "中山"
    sample = df[df['レース名'].astype(str).str.contains('25/01') & df['レース名'].astype(str).str.contains('中山')]
    
    if not sample.empty:
        print(f"Found {len(sample)} records.")
        print("Samples:")
        for i, val in enumerate(sample['レース名'].unique()[:20]):
            print(f"[{i}] '{val}'")
    else:
        print("No matches found for '25/01' and '中山'. Checking general format...")
        print(df['レース名'].head(10))
else:
    print("Column missing.")
