"""
Debug script to check horse name extraction from scraping
"""
import sys
import os
import pandas as pd

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.scraping import Shutuba
from modules.data_loader import fetch_and_process_race_data

def debug_race(race_id):
    print(f"--- Debugging Race ID: {race_id} ---")
    
    # 1. Scraping Direct Check
    print("\n[1] Checking Shutuba.scrape()...")
    df = Shutuba.scrape(race_id)
    if df.empty:
        print("Shutuba.scrape returned empty dataframe.")
    else:
        print(f"Columns: {df.columns.tolist()}")
        if '馬名' in df.columns:
            print("Horse Names found:")
            print(df[['馬番', '馬名']].head())
        else:
            print("CRITICAL: '馬名' column NOT found in scraped data.")
            
    # 2. Data Loader Check
    print("\n[2] Checking fetch_and_process_race_data()...")
    try:
        final_df = fetch_and_process_race_data(race_id)
        print(f"Final DF Columns: {final_df.columns.tolist()}")
        if '馬名' in final_df.columns:
            print("Final Horse Names:")
            print(final_df[['馬番', '馬名']].head())
        else:
             print("CRITICAL: '馬名' column NOT found in processed data.")
             
    except Exception as e:
        print(f"Error in data loader: {e}")

if __name__ == "__main__":
    # 2026年 AJCC (アメリカジョッキークラブカップ) のID: 202606010911
    # または適当なレースID
    race_id = "202606010911" 
    debug_race(race_id)
