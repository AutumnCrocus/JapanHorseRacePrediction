
import pandas as pd
import pickle
import os

RESULTS_FILE = 'data/raw/results.pickle'

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'rb') as f:
        df = pickle.load(f)
    print(f"Shape: {df.shape}")
    print(f"Index Name: {df.index.name}")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    
    if isinstance(df.index, pd.Index) and df.index.name == 'race_id':
        print(f"Index Sample: {df.index[:5].tolist()}")
    else:
        print(f"Index Sample: {df.index[:5].tolist()}")
        if 'race_id' in df.columns:
            print(f"race_id Column Sample: {df['race_id'].astype(str).head(5).tolist()}")
        else:
            print("race_id column not found in columns.")
            
    # Check 2025 data
    try:
        # race_idを特定
        if df.index.name == 'race_id':
            rids = df.index.astype(str)
        elif 'race_id' in df.columns:
            rids = df['race_id'].astype(str)
        else:
            rids = pd.Series([])
            
        rids_2025 = rids[rids.str.startswith('2025')]
        print(f"2025 Race IDs count: {len(rids_2025.unique())}")
        print(f"2025 Race IDs sample: {rids_2025.unique()[:5]}")
    except Exception as e:
        print(f"Error inspecting 2025 data: {e}")
else:
    print(f"{RESULTS_FILE} not found.")
