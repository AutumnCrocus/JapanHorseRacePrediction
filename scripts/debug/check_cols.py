import pandas as pd
import pickle
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import RAW_DATA_DIR

try:
    file_path = os.path.join(RAW_DATA_DIR, 'horse_results.pickle')
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
        print("Columns:", df.columns.tolist())
        print("Head:", df.head(1).to_dict())
except Exception as e:
    print(e)
