
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import PROCESSED_DATA_DIR

DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')

def check_data():
    print("=== Data Distribution Check ===")
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found!")
        return

    with open(DATASET_PATH, 'rb') as f:
        data = pickle.load(f)
    df = data['data']
    
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    # Check Year Column
    if 'year' not in df.columns:
        print("Warning: 'year' column missing. Inferring from race_id or date...")
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        elif 'race_id' in df.columns:
             # Assuming YYYY... format
             df['year'] = df['race_id'].astype(str).str[:4].astype(int)
        else:
             # Try index
             df = df.reset_index()
             if 'race_id' in df.columns:
                 df['year'] = df['race_id'].astype(str).str[:4].astype(int)
             elif 'index' in df.columns and df['index'].astype(str).str[:4].str.isnumeric().all():
                 df['year'] = df['index'].astype(str).str[:4].astype(int)
    
    if 'year' not in df.columns:
        print("Error: Could not determine year.")
        return

    print("\nYear Distribution:")
    print(df['year'].value_counts().sort_index())
    
    # Check horse_id
    if 'horse_id' not in df.columns:
        print("Error: horse_id column missing.")
        return
        
    train_mask = df['year'] <= 2024
    test_mask = df['year'] == 2025
    
    train_horses = set(df[train_mask]['horse_id'].dropna().unique())
    test_horses = set(df[test_mask]['horse_id'].dropna().unique())
    
    print(f"\nTrain Horses (<=2024): {len(train_horses)}")
    print(f"Test Horses (2025): {len(test_horses)}")
    
    new_horses = test_horses - train_horses
    print(f"New Horses in 2025: {len(new_horses)}")
    
    if len(new_horses) > 0:
        print("Examples of new horses:", list(new_horses)[:5])
    else:
        print("Strange: No new horses found. Check data leakage or extensive history.")

if __name__ == "__main__":
    check_data()
