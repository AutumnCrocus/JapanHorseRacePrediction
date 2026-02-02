import pandas as pd
import pickle
import os
import re

DATA_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'
RESULTS_FILE = os.path.join(DATA_DIR, 'results.pickle')

def inspect():
    print("Loading results.pickle...")
    with open(RESULTS_FILE, 'rb') as f:
        df = pickle.load(f)
    
    # Ensure race_id index/column
    if 'race_id' not in df.columns:
        df['race_id'] = df.index.astype(str)
    
    # Extract Year
    df['year'] = df['race_id'].str[:4].astype(int)
    
    # Filter 2025
    df_2025 = df[df['year'] == 2025]
    print(f"Total rows: {len(df)}")
    print(f"2025 rows: {len(df_2025)}")
    
    if df_2025.empty:
        print("No 2025 data found in results.pickle!")
        print(f"Years found: {df['year'].unique()}")
        return

    # Check Race Names for Graded Logic
    if 'レース名' in df_2025.columns:
        print("Sample Race Names:")
        print(df_2025['レース名'].head(10))
        
        # Regex for Grade
        # Common formats: "第xx回有馬記念(G1)", "～～ステークス(G3)"
        graded = df_2025[df_2025['レース名'].str.contains(r'G[1-3]|Jpn[1-3]', regex=True, na=False)]
        print(f"Estimated Graded Races (rows): {len(graded)}")
        print(f"Unique Graded Races: {len(graded['race_id'].unique())}")
        print("Sample Graded Races:")
        print(graded[['race_id', 'レース名']].drop_duplicates('race_id').head(5))
    else:
        print("'レース名' column missing!")
        print(f"Columns: {df_2025.columns.tolist()}")

if __name__ == '__main__':
    inspect()
