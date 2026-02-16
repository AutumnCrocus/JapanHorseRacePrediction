
"""
2010-2020 データセットでのモデル学習 (Refactored)
- 目的: 2021以降の長期シミュレーション用モデルを作成
- 特徴: 徹底的なデータクレンジングとType Safety
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.preprocessing import DataProcessor as RaceDataProcessor, FeatureEngineer

# Config
TRAIN_START = '2010-01-01'
TRAIN_END = '2020-12-31'
OUTPUT_DIR = os.path.join(MODEL_DIR, "historical_2010_2020")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"=== Model Training (2010-2020) Started ===", flush=True)
    
    # 1. Load Data
    print("Loading data...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
        
    # IMMEDIATE CLEANUP
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    # Deduplicate immediately
    results = results.loc[:, ~results.columns.duplicated()]
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    # 2. Filter Period
    print(f"Filtering data from {TRAIN_START} to {TRAIN_END}...", flush=True)
    df_raw = results[(results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)].copy()
    print(f"Train samples: {len(df_raw)}", flush=True)
    
    # 3. Preprocessing & Feature Engineering
    print("Processing features...", flush=True)
    processor = RaceDataProcessor()
    engineer = FeatureEngineer()
    
    # Apply processing
    import traceback
    
    # Apply processing
    try:
        print("Debug: Processing results...", flush=True)
        df_proc = processor.process_results(df_raw)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("Debug: Adding horse history...", flush=True)
        try:
            df_proc = engineer.add_horse_history_features(df_proc, hr)
        except:
            traceback.print_exc()
            print("Skipping horse history due to error", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("Debug: Adding course suitability...", flush=True)
        try:
            df_proc = engineer.add_course_suitability_features(df_proc, hr)
        except:
            traceback.print_exc()
            print("Skipping course suitability due to error", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("Debug: Adding jockey features...", flush=True)
        try:
            df_proc, _ = engineer.add_jockey_features(df_proc)
        except:
            traceback.print_exc()
            print("Skipping jockey features due to error", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("Debug: Adding pedigree features...", flush=True)
        try:
            df_proc = engineer.add_pedigree_features(df_proc, peds)
        except:
            traceback.print_exc()
            print("Skipping pedigree features due to error", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("Debug: Adding odds features...", flush=True)
        try:
            df_proc = engineer.add_odds_features(df_proc)
        except:
            traceback.print_exc()
            print("Skipping odds features due to error", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
    except Exception as e:
        print(f"Feature processing CRITICAL failure: {e}")
        traceback.print_exc()
        return

    # 4. Clean Slate Construction
    print("Constructing clean training set...", flush=True)
    
    # Prepare target
    if 'rank' not in df_proc.columns and '着順' in df_proc.columns:
        df_proc['rank'] = df_proc['着順']
        
    # Check 'rank' integrity
    if 'rank' not in df_proc.columns:
        print("Error: 'rank' column missing.")
        return
        
    rank_series = df_proc['rank']
    if isinstance(rank_series, pd.DataFrame):
        rank_series = rank_series.iloc[:, 0]
        
    target_series = rank_series.apply(lambda x: 1 if x <= 3 else 0)
    date_series = df_proc['date']
    if isinstance(date_series, pd.DataFrame):
        date_series = date_series.iloc[:, 0]
    
    # Prepare features
    # CRITICAL: Exclude post-race features (Prize, Time, Final Odds)
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 
        'time', '着差', '通過', '上り', '単勝', '人気', 
        'horse_name', 'jockey', 'trainer', 'owner', 'gender', 'original_race_id',
        '賞金（万円）', 'タイム指数', 'タイム秒', 'odds', 'popularity', 'is_win',
        'return', 'rank_num'
    ]
    
    # Reset index to avoid alignment issues
    df_proc = df_proc.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True)
    date_series = date_series.reset_index(drop=True)

    # New DataFrame
    df_clean = pd.DataFrame(index=df_proc.index)
    
    valid_features = []
    seen_cols = set()
    
    for col in df_proc.columns:
        if col in exclude_cols:
            continue
        if col in seen_cols:
            continue
        seen_cols.add(col)
            
        # Get data safely
        col_data = df_proc[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
            
        # Check numeric
        if pd.api.types.is_numeric_dtype(col_data):
            df_clean[col] = col_data
            valid_features.append(col)
        else:
            try:
                converted = pd.to_numeric(col_data, errors='coerce')
                if converted.notna().sum() > 0:
                    df_clean[col] = converted
                    valid_features.append(col)
            except:
                pass

    print(f"Valid features: {len(valid_features)}", flush=True)
    
    # Assign target and date
    df_clean['target'] = target_series
    df_clean['date'] = date_series
    
    # Inspection
    print("Debug: df_clean columns:", df_clean.columns.tolist(), flush=True)
    print("Debug: df_clean shape:", df_clean.shape, flush=True)
    if 'date' in df_clean.columns:
        print("Debug: date type:", type(df_clean['date']), flush=True)
    
    # Dropna
    prop_size = len(df_clean)
    df_clean = df_clean.dropna(subset=['target', 'date'])
    print(f"Dropped {prop_size - len(df_clean)} rows with missing target/date", flush=True)
    
    # Sort
    print("Sorting...", flush=True)
    try:
        df_clean = df_clean.sort_values('date')
    except Exception as e:
        print(f"Sort failed: {e}")
        # Critical failure debugging
        print("Debug: date column content:")
        print(df_clean['date'].head())
        return
    
    # Split
    print("Splitting...", flush=True)
    split_idx = int(len(df_clean) * 0.8)
    
    X = df_clean[valid_features]
    y = df_clean['target']
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}", flush=True)
    
    # Train
    print("Training...", flush=True)
    model = HorseRaceModel()
    model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Save
    print(f"Saving model to {OUTPUT_DIR}...", flush=True)
    model.save(os.path.join(OUTPUT_DIR, 'model.pkl'))
    
    with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(OUTPUT_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
        
    print("Training Completed Successfully.", flush=True)

if __name__ == "__main__":
    main()
