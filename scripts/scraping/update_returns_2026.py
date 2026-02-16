
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, RETURN_FILE, RESULTS_FILE
from modules.scraping import Return

def run():
    print("=== Updating Return Tables for 2026 ===")
    
    # Load Results to get Race IDs for 2026
    # We want ALL 2026 races that we have results for.
    # Note: `results.pickle` (raw) has race_id as index.
    
    if not os.path.exists(os.path.join(RAW_DATA_DIR, RESULTS_FILE)):
        print("results.pickle not found.")
        return

    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
        
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
    else:
        results['race_id'] = results.index.astype(str)
        
    # Filter 2026
    races_2026 = results[results['race_id'].str.startswith('2026')]['race_id'].unique().tolist()
    print(f"Found {len(races_2026)} races for 2026 in results.")
    
    if not races_2026:
        print("No races found.")
        return
        
    # Check existing returns
    return_path = os.path.join(RAW_DATA_DIR, RETURN_FILE)
    if os.path.exists(return_path):
        with open(return_path, 'rb') as f:
            existing_returns = pickle.load(f)
        print(f"Loaded {len(existing_returns)} existing returns.")
    else:
        existing_returns = pd.DataFrame()
        print("No existing returns found. Creating new.")

    # Filter for missing
    # Check if race_id is in index (level 0)
    if not existing_returns.empty:
        existing_rids = set(existing_returns.index.get_level_values(0).unique().astype(str))
    else:
        existing_rids = set()
        
    missing_rids = [rid for rid in races_2026 if rid not in existing_rids]
    print(f"Missing returns for {len(missing_rids)} races.")
    
    if not missing_rids:
        print("All returns already present.")
        return

    # Batch scrape (to save progress)
    print(f"Scraping {len(missing_rids)} races...")
    new_returns_df = Return.scrape(missing_rids)
    
    if not new_returns_df.empty:
        # Ensure column names match (int vs str)
        # existing columns are likely Int64Index([0, 1, 2, 3])
        # new columns might be RangeIndex(0, 3) or similar.
        new_returns_df.columns = range(new_returns_df.shape[1])
        
        # Concat
        updated_df = pd.concat([existing_returns, new_returns_df])
        
        print(f"Added {len(new_returns_df)} rows.")
        
        # Save
        with open(return_path, 'wb') as f:
            pickle.dump(updated_df, f)
        print(f"Saved updated returns to {return_path}")
        
    else:
        print("No data scraped.")


if __name__ == "__main__":
    run()
