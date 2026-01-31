
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import RAW_DATA_DIR
from modules.training import HorseRaceModel
from modules.betting_allocator import BettingAllocator
from modules.preprocessing import DataProcessor, FeatureEngineer

def main():
    print("Loading data...")
    # Load SMALL subset for speed? No, need full history for features usually.
    # But for debugging "No Bets", we can just use 2025 data if features don't crash.
    # We load full results but limit test set.
    results = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'results.pickle'))
    horse_results = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'horse_results.pickle'))
    peds = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'peds.pickle'))
    
    # Preproc
    results['year'] = results.index.astype(str).str[:4].astype(int)
    results['date'] = pd.to_datetime(results['year'].astype(str) + '-01-01')
    
    # Split
    # Taking only small amount of train to make feature eng fast?
    # We need just enough to make model work.
    
    # Focus on 2025 (Test)
    df_test_raw = results[results.index.astype(str).str.startswith('2025')].head(500) # 500 horses ~ 30 races
    df_train_raw = results.head(1000) # Small train
    
    print("Preparing Data...")
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    full_df = pd.concat([df_train_raw, df_test_raw])
    
    # Index fix
    full_df.index.name = 'race_id_index'
    full_df = full_df.reset_index()
    
    # Basic
    full_df = processor.process_results(full_df)
    
    # Skip heavy history/peds for this debug? 
    # If we skip, model performance sucks but mechanism works.
    # Let's Skip history to be super fast.
    
    # Encode
    categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    categorical_cols = [c for c in categorical_cols if c in full_df.columns]
    full_df = processor.encode_categorical(full_df, categorical_cols)
    
    # Restore Index
    if 'race_id_index' in full_df.columns:
        full_df = full_df.set_index('race_id_index')
        
    X_train = full_df.iloc[:len(df_train_raw)].copy()
    y_train = (pd.to_numeric(X_train['着順'], errors='coerce') <= 3).astype(int)
    
    # Mock Model (Random)
    class MockModel:
        def predict(self, X):
            return np.random.rand(len(X))
            
    model = MockModel()
    
    print("Starting Simulation Loop...")
    df_test = full_df.iloc[len(df_train_raw):].sort_index()
    unique_races = df_test.index.unique()
    
    log = []
    
    for rid in unique_races:
        print(f"--- Race {rid} ---")
        race_df = df_test.loc[rid]
        if isinstance(race_df, pd.Series): race_df = race_df.to_frame().T
        
        probs = model.predict(race_df)
        
        df_preds = race_df.copy()
        df_preds['probability'] = probs
        if '馬番' in df_preds.columns:
            df_preds['horse_number'] = pd.to_numeric(df_preds['馬番'], errors='coerce').fillna(0).astype(int)
        else:
            df_preds['horse_number'] = range(1, len(df_preds)+1)
            
        df_preds['odds'] = None
        df_preds['expected_value'] = 0
        df_preds['horse_name'] = df_preds['馬名'] if '馬名' in df_preds.columns else [f"H{i}" for i in df_preds['horse_number']]
        
        budget = 5000
        
        try:
            recs = BettingAllocator.allocate_budget(df_preds, budget, odds_data=None)
            print(f"Recs: {len(recs)}")
            for r in recs:
                print(f"  {r['bet_type']} {r['method']} {r['combination']}")
                log.append(r)
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Total Bets: {len(log)}")

if __name__ == "__main__":
    main()
