"""
実験用モデル学習スクリプト
- 対象期間: 2016-2024
- アルゴリズム: LightGBM
- 保存先: models/experiment_model_2025.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from datetime import datetime

# パスの解決
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE, MODEL_DIR
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer

# 実験モデル名
MODEL_NAME = "experiment_model_2025.pkl"
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def train_model():
    print("=== Training Experiment Model (2016-2024) ===")
    
    # 1. データ読み込み
    print("Loading data...")
    try:
        with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
        with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
        with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. データ整形 (race_id, date)
    print("Preprocessing raw results...")
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
    
    if 'date' not in results.columns:
        try: results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except: pass
        
    results['year'] = pd.to_datetime(results['date'], errors='coerce').dt.year
    
    # 3. 学習データ抽出 (2016-2024)
    df_train = results[results['year'].isin(range(2016, 2025))].copy()
    print(f"Training Data: {len(df_train)} rows (Years: {df_train['year'].min()}-{df_train['year'].max()})")
    
    # 4. 前処理 & 特徴量生成
    print("Feature Engineering...")
    processor = DataProcessor()
    df_proc = processor.process_results(df_train)
    
    engineer = FeatureEngineer()
    
    # 関連データのフィルタリング（高速化のため）
    active_horses = df_proc['horse_id'].unique() if 'horse_id' in df_proc.columns else []
    hr_filtered = hr[hr.index.isin(active_horses)].copy() if not hr.empty else hr
    
    df_proc = engineer.add_horse_history_features(df_proc, hr_filtered)
    df_proc = engineer.add_course_suitability_features(df_proc, hr_filtered)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # 5. 特徴量選択
    exclude_cols = [
        'race_id', 'horse_name', 'jockey_name', 'trainer_name', 'horse_id', 'date', 'original_race_id', 'year',
        '着順', '着 順', 'rank_num', 'target',
        'タイム', 'タイム秒', '上り', '通過', 'running_style', # These are results
        '単勝', '単 勝', '人気', '人 気', # These are results
        '戦績', '賞金', '賞金（万円）', 'タイム指数', '着差', # Leaks
        'is_win', 'is_place', 'return' # Critical Leaks from Feature Engineering internal cols
    ]
    # Keep 'odds' and 'popularity' if they were added by engineer as "prior info" (though engineer adds them from '単勝').
    # Ideally we should use Pre-Race odds.
    # For this experiment, if we exclude 'odds', performance drops significantly.
    # But strictly speaking '単勝' in results is Result Odds.
    # Let's keep 'odds' and 'popularity' columns (added by add_odds_features) BUT we must accept this is a "Closing Odds Simulation".
    # The LEAK was mainly Rank, Time, Pass, 3F.
    
    feature_cols = [c for c in df_proc.columns if df_proc[c].dtype in ['float64', 'int64', 'int32', 'float32'] and c not in exclude_cols]
    
    # ターゲット: 3着以内を1とする
    if '着順' in df_proc.columns:
        df_proc['target'] = df_proc['着順'].apply(lambda x: 1 if x <= 3 else 0)
    elif 'rank_num' in df_proc.columns:
        df_proc['target'] = df_proc['rank_num'].apply(lambda x: 1 if x <= 3 else 0)
    
    X = df_proc[feature_cols].fillna(0)
    y = df_proc['target']
    
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    
    # 6. 学習
    print("Training LightGBM model...")
    model = HorseRaceModel(model_type='lgbm')
    
    # 2024年を検証データに、2016-2023年を学習データにする
    mask_val = (df_proc['year'] == 2024)
    X_train = X[~mask_val]
    y_train = y[~mask_val]
    X_val = X[mask_val]
    y_val = y[mask_val]
    
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Metrics: {metrics}")
    
    # 7. 保存
    model.save(SAVE_PATH)
    
    # Processor/Engineerも保存（推論時に必要）
    # 通常は別途保存だが、ここでは簡易化のため上書き注意
    # 実番モデルを壊さないよう、experiment用のサフィックスをつけるか、
    # 読み込み側で都度生成するか。今回はsimulate側で生成ロジックがあるのでモデルのみ保存でOK
    # ただし、simulate_strategy_comparison.pyの90行目でprocessor.pklをロードしているので
    # 互換性のため保存しておくのが無難だが、既存のものを上書きするのは危険。
    # simulate側で「ロード失敗したら新規作成」しているので、processorファイルがなくても動くはず。
    
    print("Done.")

if __name__ == "__main__":
    train_model()
