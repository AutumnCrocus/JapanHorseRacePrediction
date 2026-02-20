
import os
import sys
import pandas as pd
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import PROCESSED_DATA_DIR

DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')

def inspect():
    with open(DATASET_PATH, 'rb') as f:
        data = pickle.load(f)
        df = data['data']
        print("Columns:", df.columns.tolist())
        print("Index:", df.index)
        print("Index Name:", df.index.name)
        # Check for race_id column
        if 'race_id' in df.columns:
             print("race_id column exists")
        else:
             print("race_id column MISSING")
             
        # Check if index looks like race_id
        sample_idx = df.index[0]
        print(f"Sample Index: {sample_idx} (Type: {type(sample_idx)})")
        
        # Check for similar columns
        print("Columns matching *race*:", [c for c in df.columns if 'race' in c])

if __name__ == "__main__":
    inspect()
