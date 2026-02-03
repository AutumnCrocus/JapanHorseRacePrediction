
import os
import sys
import pickle
import pandas as pd

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.analyze_2025_breakdown import create_race_info_map, load_resources
from modules.constants import DATA_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE

def debug_analysis():
    print("=== Debugging Analysis Breakdown ===")
    
    # Check 1: Results Data
    print("\n[1] Checking Results Data...")
    try:
        with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
            results = pickle.load(f)
        
        if isinstance(results.index, pd.MultiIndex):
            results = results.reset_index()
            # Assuming level 0 is race_id if it was multi-index
            if 'race_id' not in results.columns:
                results['race_id'] = results.iloc[:, 0].astype(str)
        elif 'race_id' not in results.columns:
             results['race_id'] = results.index.astype(str)
        
        results_2025 = results[results['race_id'].str.startswith('2025')]
        n_unique = results_2025['race_id'].nunique()
        n_rows = len(results_2025)
        print(f"Total rows in 2025: {n_rows}")
        print(f"Unique race_ids in 2025: {n_unique}")
        print(f"Sample race_ids: {results_2025['race_id'].head().tolist()}")
    except Exception as e:
        print(f"Error loading results: {e}")

    # Check 2: Returns Data
    print("\n[2] Checking Returns Data...")
    try:
        model, processor, engineer, bias_map, jockey_stats, returns_df = load_resources()
        print(f"Returns DF index type: {type(returns_df.index)}")
        print(f"Returns DF sample index: {returns_df.index[:5].tolist()}")
        
        # Check intersection
        if 'results_2025' in locals():
            sample_rid = results_2025['race_id'].iloc[0]
            print(f"Checking sample race_id '{sample_rid}' in returns_df...")
            if sample_rid in returns_df.index:
                print("  -> Found!")
            else:
                print("  -> NOT Found!")
    except Exception as e:
        print(f"Error loading resources: {e}")

    # Check 3: Date Map & Race Meta
    print("\n[3] Checking Race Meta Map...")
    try:
        race_meta = create_race_info_map()
        print(f"Mapped races count: {len(race_meta)}")
        if len(race_meta) > 0:
            sample_key = list(race_meta.keys())[0]
            print(f"Sample meta ({sample_key}): {race_meta[sample_key]}")
        else:
            print("Race meta is empty!")
            
        # Check Date Map specifically
        date_map_file = os.path.join(DATA_DIR, "date_map_2025.pickle")
        if os.path.exists(date_map_file):
            with open(date_map_file, 'rb') as f:
                date_map = pickle.load(f)
            print(f"Date map size: {len(date_map)}")
            print(f"Date map sample: {list(date_map.items())[:5]}")
        else:
            print("Date map file not found!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error creating race info map: {e}")

if __name__ == "__main__":
    debug_analysis()
