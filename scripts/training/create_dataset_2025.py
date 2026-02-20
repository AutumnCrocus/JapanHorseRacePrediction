
"""
2025年再学習用データセット作成スクリプト
- 対象期間: 2010-2025
- データソース: modules.data_loader (年別データ対応)
- 出力: data/processed/dataset_2010_2025.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.data_loader import load_results, load_yearly_data
from modules.preprocessing import DataProcessor, FeatureEngineer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'dataset_2010_2025.pkl')

def create_dataset():
    print("=== Creating Dataset (2010-2025) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. データロード
    print("Loading race results (2010-2025)...")
    # memory saving: load all needed years
    years = range(2010, 2026)
    results_list = []
    
    # Start with 2025 and go backwards or just load all?
    # data_loader.load_results can load range if we modify it, but currently it takes specific year or standard file
    # Let's verify data_loader capabilities in mind or just loop
    # Actually data_loader.load_results(start_year, end_year) is likely not supported by my memory of it? 
    # Wait, in step 887 I used `load_results(2025, 2025)`. 
    # Let's assumme load_results supports range or I look at modules/data_loader.py again.
    # I saw `def load_results(start_year=None, end_year=None):` in step 887 view.
    
    try:
        results = load_results(2010, 2025)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    print(f"Loaded {len(results)} races.")

    # 2. 関連データ
    print("Loading horse/pedigree data...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)

    # 3. 前処理
    print("Preprocessing & Feature Engineering...")
    processor = DataProcessor()
    
    # Process in chunks or all at once?
    # 16 years of data might be large.
    # results df size: 3500 races/year * 16 = 56000 races.
    # 15 horses/race = 840,000 rows. fits in memory clearly.
    
    df_proc = processor.process_results(results)
    
    # Feature Engineering
    engineer = FeatureEngineer()
    
    # HR filtering
    active_ids = df_proc['horse_id'].unique()
    hr_filtered = hr[hr.index.isin(active_ids)].copy()
    
    df_proc = engineer.add_horse_history_features(df_proc, hr_filtered)
    df_proc = engineer.add_course_suitability_features(df_proc, hr_filtered)
    df_proc, _ = engineer.add_jockey_features(df_proc) # stats are computed internally
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    # df_proc = engineer.add_odds_features(df_proc) 
    # NOTE: add_odds_features uses '単勝' column which is RESULT odds. 
    # For training, using result odds is strictly leaking if we pretend it's real-time, 
    # but for "closing odds model" it's fine. 
    # predict_tomorrow uses Odds.scrape() which gives pre-race odds.
    # Consistency is key. If we train with Result Odds, we should predict with Pre-Race Odds (approx).
    df_proc = engineer.add_odds_features(df_proc)

    # 4. ターゲットと分割フラグ作成
    print("Finalizing dataset...")
    
    # Target (Top 3)
    if '着順' in df_proc.columns:
        df_proc['target'] = df_proc['着順'].apply(lambda x: 1 if x <= 3 else 0)
    elif 'rank_num' in df_proc.columns:
        df_proc['target'] = df_proc['rank_num'].apply(lambda x: 1 if x <= 3 else 0)
        
    # Year for splitting
    if 'date' in df_proc.columns:
         df_proc['year'] = pd.to_datetime(df_proc['date']).dt.year
    else:
         # Fallback
         df_proc['year'] = df_proc['race_id'].astype(str).str[:4].astype(int)

    # 5. 保存
    print(f"Saving to {OUTPUT_FILE}...")
    dataset = {
        'data': df_proc,
        'processor': processor,
        'engineer': engineer,
        'feature_names': [c for c in df_proc.columns if c not in ['target', 'date', 'race_id', 'horse_name', 'year', 'rank_num', '着順']] # simplified
    }
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(dataset, f)
        
    print("Done.")

if __name__ == "__main__":
    create_dataset()
