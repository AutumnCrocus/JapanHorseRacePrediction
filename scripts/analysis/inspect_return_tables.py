import pandas as pd
import os

def inspect_return_tables():
    file_path = 'data/raw/return_tables.pickle'
    if not os.path.exists(file_path):
        print("File not found.")
        return

    print(f"Loading {file_path}...")
    try:
        df = pd.read_pickle(file_path)
        print(f"Loaded {len(df)} records.")
        print("\n--- Columns ---")
        print(df.columns.tolist())
        
        print("\n--- Index ---")
        print(df.index.names)
        
        print("\n--- Head (First 1) ---")
        pd.set_option('display.max_columns', None)
        print(df.head(1))
        
        # Check various bet types
        print("\n--- Bet Type Samples ---")
        bet_types = ['馬連', '馬単', 'ワイド', '3連複', '3連単']
        for bt in bet_types:
            sample = df[df[0] == bt].head(3)
            print(f"\n[{bt}]")
            print(sample)
            
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    inspect_return_tables()
