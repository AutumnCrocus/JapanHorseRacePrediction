
import pandas as pd
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from modules.constants import PROCESSED_DATA_DIR

def inspect():
    path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    print(f"Loading {path}...")
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        
        print("Columns:", df.columns.tolist())
        print("Head:", df.head(1).to_dict())
        
        if 'date' in df.columns:
            print("Found 'date' column.")
        elif '日付' in df.columns:
            print("Found '日付' column.")
        else:
            print("Date column NOT found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
