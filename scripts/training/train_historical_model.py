
import os
import sys
import pickle
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import (
    RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
)
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

# Config
TRAIN_START = '2010-01-01'
TRAIN_END = '2024-12-31'
HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
os.makedirs(HISTORICAL_MODEL_DIR, exist_ok=True)

def train_historical():
    print(f"=== Training Historical Model ({TRAIN_START} - {TRAIN_END}) ===")
    
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
    # Reconstruct date if missing or use year for filtering
    if 'date' not in results.columns:
        print("Warning: 'date' column missing. Using year from index for filtering.")
        # Race ID format: YYYY...
        if isinstance(results.index, pd.MultiIndex):
            years = results.index.get_level_values(0).astype(str).str[:4]
        else:
            years = results.index.astype(str).str[:4]
        results['year_tmp'] = pd.to_numeric(years, errors='coerce')
        df_train = results[(results['year_tmp'] >= 2010) & (results['year_tmp'] <= 2024)].copy()
        df_train.drop(columns=['year_tmp'], inplace=True)
    else:
        results['date'] = pd.to_datetime(results['date'], errors='coerce')
        df_train = results[(results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)].copy()
    
    print(f"Filtered Training Data: {len(df_train)} rows")
    if not df_train.empty:
        # Check years
        if isinstance(df_train.index, pd.MultiIndex):
            y_range = df_train.index.get_level_values(0).astype(str).str[:4].unique()
        else:
            y_range = df_train.index.astype(str).str[:4].unique()
        print(f"Years in training set: {sorted(list(y_range))}")
    
    if df_train.empty:
        print("No data in range.")
        return

    # 3. Preprocessing
    print("Preprocessing & Feature Engineering...")
    # prepare_training_data logic (reuses current logic)
    # Note: prepare_training_data usually splits or processes all?
    # It returns X, y, processor, etc.
    # It doesn't filter by date itself.
    
    X, y, processor, engineer, bias_map, jockey_stats, _ = prepare_training_data(
        df_train, hr, peds, scale=False
    )
    
    # Drop non-feature cols
    if 'original_race_id' in X.columns: X = X.drop(columns=['original_race_id'])
    
    print(f"Training Features: {X.shape}")
    
    # 4. Train
    print("Training LightGBM...")
    model = HorseRaceModel(model_type='lgbm')
    
    # Use last 10% of this specific period as validation to check overfitting
    metrics = model.train(X, y, test_size=0.1)
    
    print("Metrics:", metrics)
    
    # 5. Save
    print(f"Saving to {HISTORICAL_MODEL_DIR}...")
    model.save(os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl'))
    
    # Save auxiliary objects needed for inference
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'bias_map.pkl'), 'wb') as f:
        pickle.dump(bias_map, f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'jockey_stats.pkl'), 'wb') as f:
        pickle.dump(jockey_stats, f)
        
    print("Done.")

if __name__ == "__main__":
    train_historical()
