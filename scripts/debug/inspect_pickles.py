
import pandas as pd
import pickle
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def inspect_data():
    base_dir = r"c:\Users\t4kic\Documents\JapanHorseRacePrediction"
    model_dir = os.path.join(base_dir, "models")
    
    print(f"Base Dir: {base_dir}")
    
    # Check Horse Results
    hr_path = os.path.join(base_dir, "data", "raw", "horse_results.pickle")
    print(f"Checking {hr_path}...")
    if os.path.exists(hr_path):
        print(f"Loading {hr_path}...")
        try:
            df_hr = pd.read_pickle(hr_path)
            print(f"Horse Results Shape: {df_hr.shape}")
            print(f"Columns: {df_hr.columns.tolist()[:10]}...")
            
            # Check for list types in object columns
            for col in df_hr.columns:
                if df_hr[col].dtype == 'object':
                    has_list = df_hr[col].apply(lambda x: isinstance(x, list)).any()
                    if has_list:
                        print(f"WARNING: Column '{col}' contains LISTS!")
                        # Show sample
                        sample = df_hr[df_hr[col].apply(lambda x: isinstance(x, list))][col].iloc[0]
                        print(f"  Sample: {sample}")
                        
        except Exception as e:
            print(f"Error reading horse results: {e}")

    # Check Peds
    peds_path = os.path.join(base_dir, "data", "raw", "peds.pickle")
    if os.path.exists(peds_path):
        print(f"Loading {peds_path}...")
        try:
            df_peds = pd.read_pickle(peds_path)
            print(f"Peds Shape: {df_peds.shape}")
            # Check peds columns
            for col in df_peds.columns:
                if df_peds[col].dtype == 'object':
                    has_list = df_peds[col].apply(lambda x: isinstance(x, list)).any()
                    if has_list:
                        print(f"WARNING: Column '{col}' contains LISTS!")
                        sample = df_peds[df_peds[col].apply(lambda x: isinstance(x, list))][col].iloc[0]
                        print(f"  Sample: {sample}")
            
            # Check duplicate Peds indices
            dup = df_peds.index.duplicated().sum()
            print(f"Duplicate Ped indices: {dup}")
            
        except Exception as e:
            print(f"Error reading peds: {e}")

if __name__ == "__main__":
    inspect_data()
