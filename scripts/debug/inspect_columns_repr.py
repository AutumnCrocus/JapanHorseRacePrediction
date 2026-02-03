
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

print("Columns (repr):")
print([repr(c) for c in df.columns])

# Check strictly
target = '日付'
found = any(target in c for c in df.columns)
print(f"Contains '{target}' substring in any column? {found}")

# normalized check
cols_norm = df.columns.str.replace(' ', '').str.replace('\u3000', '')
print("Normalized Columns (repr):")
print([repr(c) for c in cols_norm])

if '日付' in cols_norm:
    print("Found '日付' after normalization.")
else:
    print("STILL MISSING '日付' after normalization.")
