
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

# Normalize
df.columns = df.columns.str.replace(' ', '')
print("Columns:", df.columns.tolist())

# Check for '頭数'
if '頭数' in df.columns:
    print("Found '頭数' column.")
    print("Sample:\n", df['頭数'].head(5))
else:
    print("Missing '頭数' column.")

# Check 'レース名' content
if 'レース名' in df.columns:
    print("Sample 'レース名':")
    # Show unique non-null race names (limit 20)
    print(df['レース名'].dropna().unique()[:20])
else:
    print("Missing 'レース名' column.")
