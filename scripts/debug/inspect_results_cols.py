
import pandas as pd
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from modules.constants import RAW_DATA_DIR, RESULTS_FILE

file_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
print(f"Loading {file_path}...")

with open(file_path, 'rb') as f:
    df = pickle.load(f)

print("Columns:", df.columns.tolist())
print("-" * 20)
print("First 5 rows:")
print(df.head(5))
print("-" * 20)
if isinstance(df.index, pd.MultiIndex):
    print("Index Levels:", df.index.names)
    print("Sample Indices:", df.index[:5])
else:
    print("Index Name:", df.index.name)
    print("Sample Indices:", df.index[:5])
