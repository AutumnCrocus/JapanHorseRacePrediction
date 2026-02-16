
import os
import sys
import pickle
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import DataProcessor, FeatureEngineer

def main():
    print("データをロード中...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)

    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)

    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 最初の数件でテスト
    test_df = results.head(100).copy()
    
    print("前処理実行...")
    df_proc = processor.process_results(test_df)
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    
    # 除外カラムリスト
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 'relevance',
        'time', '着差', '通過', '上り', '単勝', '人気', 'horse_name', 'jockey', 
        'trainer', 'owner', 'gender', 'original_race_id'
    ]
    
    all_numeric_cols = [c for c in df_proc.columns if pd.api.types.is_numeric_dtype(df_proc[c])]
    features = [c for c in all_numeric_cols if c not in exclude_cols]
    
    print("\n=== 使用されている数値特徴量候補 ===")
    print(", ".join(sorted(features)))
        
    print("\n=== 想定されるリークカラムのチェック ===")
    leak_vulnerables = ['タイム秒', '着差', '上り', '賞金', '通過', 'running_style', '体重変化', '体重', '斤量', '着順']
    for v in leak_vulnerables:
        if v in features:
            print(f" [!] LEAK ALERT: {v} is in features!")
        elif v in df_proc.columns:
            # Check if it was meant to be excluded but wasn't
            pass

if __name__ == "__main__":
    main()
