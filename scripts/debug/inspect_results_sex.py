
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

file_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
print(f"Loading {file_path}...")

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print("Columns:", df.columns.tolist())
print("Head:\n", df.head(3))

# Check distinguishing sex
# Usually '性齢' (e.g., '牡3', '牝4') or '性'
cols = df.columns.tolist()
sex_col = next((c for c in cols if '性' in c), None)
if sex_col:
    print(f"Found sex column: {sex_col}")
    print(df[sex_col].unique()[:20])
else:
    print("No sex column found.")
