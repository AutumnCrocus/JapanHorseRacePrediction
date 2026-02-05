
import os
import sys
import pickle
import pandas as pd
import json
import time
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import (
    RAW_DATA_DIR, RESULTS_FILE, RETURN_FILE, HORSE_RESULTS_FILE, PEDS_FILE
)
from modules.scraping import Results, Return, HorseResults, Peds, get_race_id_list

# Configuration
YEARS_TO_SCRAPE = list(range(2010, 2027)) # 2010-2026
BACKUP_DIR = os.path.join(RAW_DATA_DIR, "historical_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

def load_pickle_safe(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    return None

def save_pickle_safe(obj, path):
    # Atomic write
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)

def main():
    print("=== Historical Data Scraping (2010-2026) ===")
    
    # 1. Load Existing Data
    print("\n[1/4] Loading existing datasets...")
    results = load_pickle_safe(os.path.join(RAW_DATA_DIR, RESULTS_FILE))
    if results is None: results = pd.DataFrame()
    
    print(f"Current Results: {len(results)} rows")
    
    existing_race_ids = set()
    if not results.empty:
        if isinstance(results.index, pd.MultiIndex):
            # Assuming level 0 is race_id
            existing_race_ids = set(results.index.get_level_values(0).unique())
        else:
            existing_race_ids = set(results.index.unique())
    
    print(f"Existing Races: {len(existing_race_ids)}")
    
    # 2. Phase 1: Scrape Results
    print("\n[2/4] Phase 1: Scraping Race Results...")
    new_results_list = []
    
    for year in YEARS_TO_SCRAPE:
        # Check if year is already mostly covered
        # Count existing IDs starting with year
        year_str = str(year)
        existing_in_year = [rid for rid in existing_race_ids if str(rid).startswith(year_str)]
        
        # Estimate total races (approx 3400 per year)
        # If we have > 3000, assuming it's done for past years.
        # For 2026 (current year), always try to update.
        if year < 2026 and len(existing_in_year) > 3000:
            print(f"Year {year} seems complete ({len(existing_in_year)} races). Skipping.")
            continue
            
        print(f"Scraping Year {year} (Existing: {len(existing_in_year)})...")
        
        # Generate ID list
        target_ids = get_race_id_list(year, year)
        # Filter out existing
        to_scrape = [rid for rid in target_ids if rid not in existing_race_ids]
        
        if not to_scrape:
            print(f"No new IDs to scrape for {year}.")
            continue
            
        print(f"Targeting {len(to_scrape)} new races for {year}.")
        
        # Scrape in chunks
        chunk_size = 100
        for i in range(0, len(to_scrape), chunk_size):
            chunk = to_scrape[i:i+chunk_size]
            df = Results.scrape(chunk)
            if not df.empty:
                new_results_list.append(df)
                
                # Intermediate save to avoid total loss
                # (Simple append to main df in memory, actual save later)
                pass
            
            # Progress log
            if (i // chunk_size) % 5 == 0:
                print(f"  Progress: {i}/{len(to_scrape)}...")
        
        # Save after each year
        if new_results_list:
            print(f"Saving Year {year} results...")
            year_df = pd.concat(new_results_list)
            if results.empty:
                results = year_df
            else:
                results = pd.concat([results, year_df])
                # Do NOT drop duplicates by index (race_id) as it removes horses!
                # If we need deduplication, we must use [index, horse_id]
                # data = data.reset_index()
                # data = data.drop_duplicates(subset=['index', 'horse_id'])
                # data = data.set_index('index')
                # For now, just trust yearly batches don't overlap.
            
            save_pickle_safe(results, os.path.join(RAW_DATA_DIR, RESULTS_FILE))
            print(f"Saved Results. Total: {len(results)} rows.")
            new_results_list = [] # Clear for next year

    # Merge Results (Final check output)
    if not results.empty:
        print(f"Final Results Count: {len(results)}")

    # Refresh existing_race_ids
    if isinstance(results.index, pd.MultiIndex):
        current_race_ids = set(results.index.get_level_values(0).unique())
    else:
        current_race_ids = set(results.index.unique())

    # 3. Phase 2: Scrape Returns
    print("\n[3/4] Phase 2: Scraping Return Tables...")
    returns = load_pickle_safe(os.path.join(RAW_DATA_DIR, RETURN_FILE))
    if returns is None: 
        returns = pd.DataFrame() if isinstance(results, pd.DataFrame) else {} # Check return format. 
        # Actually standard return file is likely DataFrame or Dict.
        # Based on previous check, it is DataFrame with MultiIndex.
        returns = pd.DataFrame()

    existing_return_ids = set()
    if isinstance(returns, pd.DataFrame) and not returns.empty:
         existing_return_ids = set(returns.index.get_level_values(0).unique())
    elif isinstance(returns, dict):
         existing_return_ids = set(returns.keys())
         
    # Identify missing
    missing_returns = [rid for rid in current_race_ids if rid not in existing_return_ids]
    print(f"Missing returns for {len(missing_returns)} races.")
    
    if missing_returns:
        # Scrape
        new_returns_df = Return.scrape(missing_returns)
        if not new_returns_df.empty:
            # Merge
            # If returns is dataframe
            if isinstance(returns, pd.DataFrame):
                returns = pd.concat([returns, new_returns_df])
            else:
                # If it was dict (unlikely based on recent checks but possible if empty)
                returns = new_returns_df
                
            save_pickle_safe(returns, os.path.join(RAW_DATA_DIR, RETURN_FILE))
            print(f"Saved Returns. Total entries: {len(returns)}")
    
    # 4. Phase 3: Scrape Horse/Peds
    print("\n[4/4] Phase 3: Scraping Horse & Pedigree Data...")
    
    # Extract unique horse IDs from results
    if 'horse_id' in results.columns:
        all_horse_ids = set(results['horse_id'].dropna().astype(str).unique())
    else:
        print("Error: 'horse_id' column not found in results.")
        all_horse_ids = set()
        
    print(f"Total Unique Horses: {len(all_horse_ids)}")
    
    # Load existing
    horse_results = load_pickle_safe(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
    if horse_results is None: horse_results = pd.DataFrame()
    
    peds = load_pickle_safe(os.path.join(RAW_DATA_DIR, PEDS_FILE))
    if peds is None: peds = pd.DataFrame()
    
    # Check coverage
    # Horse Results index is horse_id? Or column?
    # Usually index.
    existing_hr_ids = set()
    if not horse_results.empty:
        existing_hr_ids = set(horse_results.index.unique().astype(str))
        
    existing_ped_ids = set()
    if not peds.empty:
        existing_ped_ids = set(peds.index.unique().astype(str))
        
    missing_hr = [hid for hid in all_horse_ids if hid not in existing_hr_ids]
    missing_ped = [hid for hid in all_horse_ids if hid not in existing_ped_ids]
    
    print(f"Missing Horse Results: {len(missing_hr)}")
    print(f"Missing Pedigrees: {len(missing_ped)}")
    
    # Scrape Horse Results
    if missing_hr:
        print("Scraping Horse Results...")
        # Chunking
        chunk_size = 500
        for i in range(0, len(missing_hr), chunk_size):
            chunk = missing_hr[i:i+chunk_size]
            df = HorseResults.scrape(chunk)
            if not df.empty:
                horse_results = pd.concat([horse_results, df])
                # Save periodically
                save_pickle_safe(horse_results, os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
            print(f"  HR Progress: {min(i+chunk_size, len(missing_hr))}/{len(missing_hr)}")
    
    # Scrape Peds
    if missing_ped:
        print("Scraping Pedigrees...")
        chunk_size = 500
        for i in range(0, len(missing_ped), chunk_size):
            chunk = missing_ped[i:i+chunk_size]
            df = Peds.scrape(chunk)
            if not df.empty:
                peds = pd.concat([peds, df])
                save_pickle_safe(peds, os.path.join(RAW_DATA_DIR, PEDS_FILE))
            print(f"  Ped Progress: {min(i+chunk_size, len(missing_ped))}/{len(missing_ped)}")

    print("\n=== All Scrapes Completed ===")

if __name__ == "__main__":
    main()
