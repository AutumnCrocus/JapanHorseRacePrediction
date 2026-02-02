import pandas as pd
import pickle
import os

DATA_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'
RETURN_FILE = os.path.join(DATA_DIR, 'return_tables.pickle')

def inspect():
    print("Loading return_tables.pickle...")
    with open(RETURN_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, pd.DataFrame):
        print(f"Columns: {data.columns.tolist()}")
        print(data.head(1))
    elif isinstance(data, dict):
        print(f"Keys sample: {list(data.keys())[:5]}")
        sample_key = list(data.keys())[0]
        print(f"Sample value for {sample_key}:")
        print(data[sample_key])
        
        # Check if race name is in value (usually DataFrame or list)
        # If DataFrame, check columns.

if __name__ == '__main__':
    inspect()
