
import pandas as pd
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from modules.constants import RAW_DATA_DIR, HORSE_RESULTS_FILE

file_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
print(f"Loading {file_path}...")

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print("Columns:", df.columns.tolist())
print("-" * 20)
print("First 5 rows '開催' or 'レース名':")
cols_to_show = [c for c in df.columns if '開催' in c]
if cols_to_show:
    print(f"Found columns: {cols_to_show}")
    print(df[cols_to_show].drop_duplicates().head(10))
else:
    print("No '開催' column found.")
    print("Columns are:", df.columns.tolist())
