
import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

def main():
    print("Loading results file...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
        
    print(f"Type: {type(results)}")
    if isinstance(results, pd.DataFrame):
        print(f"Total Columns: {len(results.columns)}")
        print(f"Columns list: {results.columns.tolist()}")
        
        # 配当関連カラムの候補
        payout_cols = [c for c in results.columns if any(x in c for x in ['単勝', '複勝', '馬連', '枠連', 'ワイド', '3連', '３連'])]
        print(f"Payout Columns found: {payout_cols}")
        
        if payout_cols:
            print("\nSample Payout Data:")
            print(results[payout_cols].head(5).to_string())
            
            # データ型の確認
            for c in payout_cols:
                print(f"{c}: {results[c].dtype}")
    else:
        print("Not a DataFrame")

if __name__ == "__main__":
    main()
