
import os
import sys
import pandas as pd
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from scripts.predict_tomorrow import load_prediction_pipeline, load_historical_data

def debug_race(race_id):
    print(f"Debugging Race ID: {race_id}")
    
    # 1. Scrape Shutuba
    print("Scraping Shutuba...")
    df_shutuba = Shutuba.scrape(race_id)
    print("Shutuba Columns:", df_shutuba.columns.tolist())
    
    # Check '単勝' column content
    if '単勝' in df_shutuba.columns:
        print("First 5 'Odds' values:")
        for i, val in enumerate(df_shutuba['単勝'].head()):
            print(f"  {i}: {val} (Type: {type(val)})")
            
    # Check '人気' column content
    if '人気' in df_shutuba.columns:
        print("First 5 'Pop' values:")
        for i, val in enumerate(df_shutuba['人気'].head()):
             print(f"  {i}: {val} (Type: {type(val)})")

    # Cleaner Logic (Proposed Fix)
    print("Running Cleaner...")
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            print(f"WARNING: Found LIST in column '{col}'!")
            print("  Values:", df_shutuba[df_shutuba[col].apply(lambda x: isinstance(x, list))][col].head())
            
            # Apply cleaning
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)
            print(f"  Cleaned '{col}'. New Type sample: {type(df_shutuba[col].iloc[0])}")

    # Add Date
    df_shutuba['date'] = pd.to_datetime('2026-01-31')

    # Load Pipeline
    predictor = load_prediction_pipeline()
    
    # Process
    print("Running process_results...")
    try:
        df_processed = predictor.processor.process_results(df_shutuba)
        print("process_results SUCCESS")
        print("Processed Columns:", df_processed.columns.tolist())
    except Exception:
        print("process_results FAILED")
        traceback.print_exc()
        return

    # Load History
    horse_results, peds = load_historical_data()
    
    # Feature Engineering steps
    print("Adding History Features...")
    try:
        df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results)
        print("History SUCCESS")
    except Exception:
        print("History FAILED")
        traceback.print_exc()
        # Continue? No point

    # Check Peds
    if peds is not None:
        print("Peds Loaded. Shape:", peds.shape)
    else:
        print("Peds is NONE!")

    print("Adding Pedigree Features...")
    try:
        if peds is not None:
            df_processed = predictor.engineer.add_pedigree_features(df_processed, peds)
            print("Pedigree SUCCESS")
    except Exception:
        print("Pedigree FAILED")
        traceback.print_exc()

if __name__ == "__main__":
    debug_race('202605010107')
