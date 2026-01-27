import sys
import os
import pandas as pd
import numpy as np

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data

def debug_alignment():
    print("Loading data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path)
    
    print(f"Results head index: {results_df.index[:3].tolist()}")
    
    # Run preprocessing
    print("Running prepare_training_data...")
    X, y, processor, engineer, bias_map, jockey_stats, df = prepare_training_data(results_df, hr_df)
    
    print("\nCheck df_full properties:")
    print(f"Index type: {type(df.index)}")
    print(f"Index head: {df.index[:3].tolist()}")
    
    check_cols = ['original_race_id', 'date', 'race_num', '馬番']
    avail = [c for c in check_cols if c in df.columns]
    print("\ndf head (check columns):")
    print(df[avail].head())
    
    if 'date' in df.columns:
        print("\nDate year counts:")
        print(df['date'].dt.year.value_counts().sort_index())
        
    print("\ny counts (if target exists):")
    print(y.value_counts())

if __name__ == '__main__':
    debug_alignment()
