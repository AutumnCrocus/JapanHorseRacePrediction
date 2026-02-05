
import pandas as pd
import pickle
import os
import sys
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

BACKUP_DIR = os.path.join(RAW_DATA_DIR, "recovery_backups")
TARGET_FILE = os.path.join(RAW_DATA_DIR, RESULTS_FILE)

def restore():
    print("=== Restoring Results from Backups ===")
    
    # 1. Load backups 2016-2026
    dfs = []
    for year in range(2016, 2027):
        fname = f"results_{year}.pickle"
        path = os.path.join(BACKUP_DIR, fname)
        if os.path.exists(path):
            print(f"Loading {fname}...")
            with open(path, 'rb') as f:
                df = pickle.load(f)
            dfs.append(df)
        else:
            print(f"Warning: {fname} missing")

    if not dfs:
        print("No backups found.")
        return

    # 2. Check 2010-2015
    # If I scraped them partially to results.pickle?
    # results.pickle currently is 13MB (maybe 2010?).
    # Let's load current results.pickle and check years.
    current_years = []
    if os.path.exists(TARGET_FILE):
        try:
            with open(TARGET_FILE, 'rb') as f:
                current_df = pickle.load(f)
            # Check year from index (YYYY)
            idx_str = current_df.index.astype(str)
            found_years = set(idx_str.str[:4].unique())
            print(f"Current results.pickle contains years: {found_years}")
            
            # If 2010 is there and size looks OK?
            # 55k rows for 2010 is plausible.
            # But earlier potential bug dropped some rows?
            # Best to re-scrape 2010 to be safe since I had the BUG in dup check.
            # The bug was: `results = results[~results.index.duplicated(keep='last')]`
            # This TRUNCATED the data to 1 row per race.
            # So the 2010 data in results.pickle is CORRUPTED (missing horses).
            print("Discarding current results.pickle due to known corruption (truncated horses).")
            
        except Exception as e:
            print(f"Error reading current results: {e}")

    # 3. Merge Backups (2016-2026)
    full_df = pd.concat(dfs)
    print(f"Merged Backups: {len(full_df)} rows")
    
    # 4. Save
    print(f"Saving to {TARGET_FILE}...")
    with open(TARGET_FILE, 'wb') as f:
        pickle.dump(full_df, f)
    print("Restore complete.")

    # 5. Trigger Scrape for 2010-2015
    # I won't do it here, but next step is to run scrape_historical_full.py
    # Since existing races (2016-2026) are restored, scrape_historical_full.py 
    # (which checks for existence) should skip them and focus on 2010-2015!
    
if __name__ == "__main__":
    restore()
