"""
時系列重み付け学習モデル訓練スクリプト
- 線形重み: 2010年=1, 2024年=15
- 学習対象: 2010-2024
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import (
    RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
)
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

# Config
TRAIN_START_YEAR = 2010
TRAIN_END_YEAR = 2024
WEIGHTED_MODEL_DIR = os.path.join(MODEL_DIR, "weighted_2010_2024")
os.makedirs(WEIGHTED_MODEL_DIR, exist_ok=True)

def calculate_linear_weights(years: pd.Series) -> np.ndarray:
    """
    線形重み計算: 2010年=1.0, 2024年=15.0
    """
    weights = years - (TRAIN_START_YEAR - 1)  # 2010 -> 1, 2024 -> 15
    weights = weights.astype(float)
    # 正規化 (平均を1に)
    weights = weights / weights.mean()
    return weights.values

def train_weighted():
    print(f"=== Training Time-Weighted Model ({TRAIN_START_YEAR} - {TRAIN_END_YEAR}) ===")
    
    # 1. Load Data
    print("Loading data...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
        
    print(f"Total Results: {len(results)}")
    
    # 2. Filter Date Range
    if 'date' not in results.columns:
        print("Warning: 'date' column missing. Using year from index for filtering.")
        if isinstance(results.index, pd.MultiIndex):
            years = results.index.get_level_values(0).astype(str).str[:4]
        else:
            years = results.index.astype(str).str[:4]
        results['year_tmp'] = pd.to_numeric(years, errors='coerce')
        df_train = results[(results['year_tmp'] >= TRAIN_START_YEAR) & (results['year_tmp'] <= TRAIN_END_YEAR)].copy()
    else:
        results['date'] = pd.to_datetime(results['date'], errors='coerce')
        results['year_tmp'] = results['date'].dt.year
        df_train = results[(results['year_tmp'] >= TRAIN_START_YEAR) & (results['year_tmp'] <= TRAIN_END_YEAR)].copy()
    
    print(f"Filtered Training Data: {len(df_train)} rows")
    if not df_train.empty:
        y_range = sorted(df_train['year_tmp'].dropna().unique().astype(int))
        print(f"Years in training set: {y_range}")
    
    if df_train.empty:
        print("No data in range.")
        return

    # 3. Preprocessing
    print("Preprocessing & Feature Engineering...")
    X, y, processor, engineer, bias_map, jockey_stats, _ = prepare_training_data(
        df_train, hr, peds, scale=False
    )
    
    # Drop non-feature cols
    if 'original_race_id' in X.columns: X = X.drop(columns=['original_race_id'])
    
    print(f"Training Features: {X.shape}")
    
    # 4. Calculate Sample Weights
    print("Calculating sample weights...")
    # prepare_training_data 後は original_race_id 列がある場合がある
    if 'original_race_id' in X.columns:
        years_from_col = X['original_race_id'].astype(str).str[:4].astype(int)
        X = X.drop(columns=['original_race_id'])
    else:
        # X.index が RangeIndex の場合、df_train の年情報を位置で紐付け
        # prepare_training_data は df_train から欠損等で一部行を落とす可能性があるため
        # 長さが一致すると仮定して位置ベースで取得
        if len(X) == len(df_train):
            years_from_col = df_train['year_tmp'].values
        else:
            # 長さが異なる場合、X の長さに合わせて df_train から取得
            print(f"Warning: X length ({len(X)}) != df_train length ({len(df_train)})")
            # 年の平均値をフォールバック
            years_from_col = np.full(len(X), 2017)
    
    years_series = pd.Series(years_from_col)
    sample_weights = calculate_linear_weights(years_series)
    print(f"Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    print(f"Weight mean: {sample_weights.mean():.2f}")
    print(f"Weight length: {len(sample_weights)}, X length: {len(X)}")
    
    # 5. Train
    print("Training LightGBM with sample weights...")
    model = HorseRaceModel(model_type='lgbm')
    
    # Use last 10% of this specific period as validation
    metrics = model.train(X, y, test_size=0.1, sample_weight=sample_weights)
    
    print("Metrics:", metrics)
    
    # 6. Save
    print(f"Saving to {WEIGHTED_MODEL_DIR}...")
    model.save(os.path.join(WEIGHTED_MODEL_DIR, 'model.pkl'))
    
    with open(os.path.join(WEIGHTED_MODEL_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(WEIGHTED_MODEL_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
    with open(os.path.join(WEIGHTED_MODEL_DIR, 'bias_map.pkl'), 'wb') as f:
        pickle.dump(bias_map, f)
    with open(os.path.join(WEIGHTED_MODEL_DIR, 'jockey_stats.pkl'), 'wb') as f:
        pickle.dump(jockey_stats, f)
        
    print("Done.")

if __name__ == "__main__":
    train_weighted()
