
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, HORSE_RESULTS_FILE

file_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
print(f"Loading {file_path}...")

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print("Normalizing columns...")
df.columns = df.columns.str.replace(' ', '')

if 'レース名' in df.columns:
    print("Checking 'レース名' samples:")
    unique_names = df['レース名'].dropna().unique()
    
    # Check for '歳' variants
    has_sai = [n for n in unique_names if '歳' in str(n)]
    print(f"Count with '歳': {len(has_sai)} / {len(unique_names)}")
    if has_sai:
        print("Sample with '歳':", has_sai[:10])
    
    # Check for '牝' variants
    has_hin = [n for n in unique_names if '牝' in str(n)]
    print(f"Count with '牝': {len(has_hin)} / {len(unique_names)}")
    if has_hin:
        print("Sample with '牝':", has_hin[:10])
        
    # Check random samples
    print("Random Samples:", unique_names[:20])

else:
    print("No 'レース名' column.")
