import pickle
import pandas as pd
import os

RAW_DATA_DIR = 'data/raw'
HORSE_RESULTS_FILE = 'horse_results.pickle'
RESULTS_FILE = 'results.pickle'

def check_leak_final():
    print("Loading horse_results.pickle...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    print(f"Loaded {len(hr)} history records.")

    print("\nLoading results.pickle...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        df_all = pickle.load(f)
    print(f"Loaded {len(df_all)} target races.")

    # hr index is horse_id. Columns include 'レース名'
    # Features are created from 'rank_num' (着順)
    
    # 2024 records from results.pickle
    df_2024 = df_all[df_all.index.astype(str).str.startswith('2024')]
    
    overlap_found = 0
    for i in range(min(500, len(df_2024))):
        race_id = df_2024.index[i]
        horse_id = df_2024['horse_id'].iloc[i]
        
        if horse_id in hr.index:
            history = hr.loc[[horse_id]] # This returns a DataFrame for that horse
            date_short = str(race_id)[2:4] + "/" + str(race_id)[4:6] + "/" + str(race_id)[6:8]
            
            # Check if this horse has a record for the same day in history
            match = history[history['レース名'].str.contains(date_short, na=False)]
            if not match.empty:
                overlap_found += 1
                if overlap_found == 1:
                    print(f"\n--- LEAK CONFIRMED ---")
                    print(f"Race ID: {race_id}, Horse ID: {horse_id}")
                    print(f"History entry found for same date:")
                    print(match[['レース名', '着順']])
                    print("\nSince FeatureEngineer.add_horse_history_features uses shift(1) after sorting by date,")
                    print("and hr rows come BEFORE df rows in the sort (due to concat order),")
                    print("the result of the current race is visible to the model as features.")
                    
    print(f"\nChecked 500 races, found {overlap_found} overlaps.")
    if overlap_found > 0:
        print("Conclusion: DATA LEAK IS PRESENT in the simulation environment.")
    else:
        print("Conclusion: No direct overlap found in sample. Data might be clean.")

if __name__ == "__main__":
    check_leak_final()
