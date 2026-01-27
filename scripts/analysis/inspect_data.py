
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import RAW_DATA_DIR, HORSE_RESULTS_FILE

def inspect():
    path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    if not os.path.exists(path):
        print("Data not found.")
        return
        
    df = pd.read_pickle(path)
    print("Columns:", df.columns.tolist())
    print("\nHead(3):")
    print(df.head(3))
    
    # オッズ、単勝、などのカラムがあるか
    cols = ['単勝', '人気', '着順', '賞金']
    for c in cols:
        if c in df.columns:
            print(f"\nExample {c}:", df[c].dropna().head(5).tolist())
            
if __name__ == '__main__':
    inspect()
