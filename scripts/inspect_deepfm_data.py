import pickle
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\models\deepfm\deepfm_data.pkl'

if not os.path.exists(DATA_FILE):
    print("File not found")
    exit()

with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)
    print("Keys:", data.keys())

    if 'feature_config' in data:
        print("Feature Config (Sparse):")
        for feat in data['feature_config']['sparse']:
            print(f"  - {feat['name']} (vocab: {feat['vocabulary_size']})")
            
    if 'train_df' in data:
        df = data['train_df']
        print(f"train_df exists. Shape: {df.shape}")
        print("train_df Columns:", df.columns.tolist())
        print(df.head())
    else:
        print("train_df MISSING from pickle.")

    if 'test_df' in data:
        print(f"test_df exists. Shape: {data['test_df'].shape}")
