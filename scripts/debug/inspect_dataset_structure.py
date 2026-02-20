
import pickle
import pandas as pd
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules import preprocessing # Ensure classes are available

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset_2010_2025.pkl')

print(f"Loading {DATASET_PATH}...")
with open(DATASET_PATH, 'rb') as f:
    data = pickle.load(f)
    
df = data['data']
with open('cols.txt', 'w', encoding='utf-8') as f:
    f.write(str(df.columns.tolist()))
    f.write("\n")
    if 'original_race_id' in df.columns:
        sample = df['original_race_id'].astype(str).head(5).tolist()
        f.write(f"original_race_id Sample: {sample}\n")
