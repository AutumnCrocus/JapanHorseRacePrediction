
import os
import sys
import pickle
import pandas as pd
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from modules.constants import DATA_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE

def inspect_metadata():
    print("=== Inspecting Metadata ===")
    
    # 1. Horse Results
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    if os.path.exists(hr_path):
        with open(hr_path, 'rb') as f:
            hr = pickle.load(f)
        hr.columns = hr.columns.str.replace(' ', '')
        print(f"Horse Results rows: {len(hr)}")
        print("Sample 'レース名' values:")
        print(hr['レース名'].head(10))
        
        # Test Regex
        sample = hr['レース名'].iloc[0]
        # Regex from analyze_2025_breakdown.py
        # r'(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\d+)R'
        regex = r'(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\d+)R'
        match = re.search(regex, str(sample))
        if match:
            print(f"Regex Match on '{sample}': {match.groups()}")
        else:
            print(f"Regex FAILED on '{sample}'")
            
    else:
        print("Horse results not found")

    # 2. Date Map
    dm_path = os.path.join(DATA_DIR, "date_map_2025.pickle")
    if os.path.exists(dm_path):
        with open(dm_path, 'rb') as f:
            dm = pickle.load(f)
        print(f"Date Map size: {len(dm)}")
        print(f"Date Map Sample: {list(dm.items())[:5]}")
    else:
        print("Date map not found")

if __name__ == "__main__":
    inspect_metadata()
