import pandas as pd
import pickle
import os

DATA_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'
RESULTS_FILE = os.path.join(DATA_DIR, 'results.pickle')

def inspect():
    print("Loading results.pickle...")
    with open(os.path.join(DATA_DIR, 'results.pickle'), 'rb') as f:
        df = pickle.load(f)
    print(f"Columns: {df.columns.tolist()}")
    if 'date' in df.columns:
        print("Column 'date' exists.")
    elif '日付' in df.columns:
        print("Column '日付' exists.")
    else:
        print("Date/日付 MISSING.")

if __name__ == '__main__':
    inspect()
