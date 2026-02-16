
import pandas as pd
import os

PRED_FILE = 'data/processed/prediction_2025_ltr.csv'

if os.path.exists(PRED_FILE):
    df = pd.read_csv(PRED_FILE)
    if 'race_id' in df.columns:
        # 文字列化
        df['race_id'] = df['race_id'].astype(str)
        print(f"Shape: {df.shape}")
        print(f"Unique RaceIDs: {df['race_id'].nunique()}")
        
        # head/tail
        rids = sorted(df['race_id'].unique().tolist())
        print(f"Head 5 RaceIDs: {rids[:5]}")
        print(f"Tail 5 RaceIDs: {rids[-5:]}")
        
        # Check specific ID
        target = '202501010101'
        if target in rids:
            print(f"Target {target} FOUND in prediction data.")
        else:
            print(f"Target {target} NOT FOUND in prediction data.")
            
            # 部分一致検索
            matches = [r for r in rids if r.startswith('20250101')]
            print(f"IDs starting with 20250101: {matches[:10]}")
    else:
        print("race_id column not found.")
else:
    print(f"{PRED_FILE} not found.")
