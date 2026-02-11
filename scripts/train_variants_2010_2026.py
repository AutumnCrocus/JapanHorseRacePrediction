
"""
2010-2026 データセットでのモデル学習バリアント作成
- 目的: 既存モデルのパラメータを調整した別バリアントを作成
- 学習期間: 2010/01/01 ~ 2026/02/08
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor as RaceDataProcessor, FeatureEngineer

# ============= 設定 =============
TRAIN_START = '2010-01-01'
TRAIN_END = '2026-02-08'

# 定義するバリアント
VARIANTS = {
    "historical_lgbm_tuned": {
        "model_type": "lgbm",
        "params": {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 63,
            "learning_rate": 0.01,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1
        }
    },
    "historical_rf": {
        "model_type": "rf",
        "params": {
            "n_estimators": 100,
            "max_depth": 12,
            "random_state": 42,
            "n_jobs": -1
        }
    }
}

def load_data():
    print("データを読み込み中...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
        
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    results = results.loc[:, ~results.columns.duplicated()]
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    return results, hr, peds

def preprocess(results, hr, peds):
    print(f"期間フィルタリング: {TRAIN_START} ~ {TRAIN_END}", flush=True)
    df_raw = results[(results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)].copy()
    
    print("特徴量を生成中...", flush=True)
    processor = RaceDataProcessor()
    engineer = FeatureEngineer()
    
    df_proc = processor.process_results(df_raw)
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # クリーンアップ
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 
        'time', '着差', '通過', '上り', '単勝', '人気', 
        'horse_name', 'jockey', 'trainer', 'owner', 'gender', 'original_race_id',
        '賞金（万円）', 'タイム指数', 'タイム秒', 'odds', 'popularity', 'is_win',
        'return', 'rank_num'
    ]
    
    target_series = df_proc['着順'].apply(lambda x: 1 if x <= 3 else 0)
    date_series = df_proc['date']
    
    valid_features = []
    for col in df_proc.columns:
        if col in exclude_cols: continue
        col_data = df_proc[col]
        if pd.api.types.is_numeric_dtype(col_data):
            valid_features.append(col)
            
    df_clean = df_proc[valid_features].copy()
    df_clean['target'] = target_series
    df_clean['date'] = date_series
    df_clean = df_clean.dropna(subset=['target', 'date']).sort_values('date')
    
    return df_clean, valid_features, processor, engineer

def main():
    results, hr, peds = load_data()
    df_clean, features, processor, engineer = preprocess(results, hr, peds)
    
    split_idx = int(len(df_clean) * 0.8)
    X_train = df_clean[features].iloc[:split_idx]
    y_train = df_clean['target'].iloc[:split_idx]
    X_val = df_clean[features].iloc[split_idx:]
    y_val = df_clean['target'].iloc[split_idx:]
    
    results_summary = []
    
    for v_name, config in VARIANTS.items():
        print(f"\n--- 学習開始: {v_name} ({config['model_type']}) ---", flush=True)
        output_dir = os.path.join(MODEL_DIR, v_name)
        os.makedirs(output_dir, exist_ok=True)
        
        model = HorseRaceModel(model_type=config['model_type'], model_params=config['params'])
        metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        
        print(f"メトリクス: {metrics}", flush=True)
        results_summary.append({"variant": v_name, **metrics})
        
        # 保存
        model.save(os.path.join(output_dir, 'model.pkl'))
        with open(os.path.join(output_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(processor, f)
        with open(os.path.join(output_dir, 'engineer.pkl'), 'wb') as f:
            pickle.dump(engineer, f)
            
    print("\n" + "="*50)
    print("学習サマリー:")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
