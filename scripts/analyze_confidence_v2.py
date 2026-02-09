"""
2025年 自信度分布解析スクリプト v2
- simulate_2025_confidence_sanrenpuku.py をベースにした高速版
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm
from collections import defaultdict

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor

# --- 設定 ---
MODEL_BASE_DIR = os.path.join('models', 'historical_2010_2024')
MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'model.pkl')
BATCH_SIZE = 50

def load_resources():
    print("Loading resources...", flush=True)
    import gc
    
    # 1. レース結果データのロードと2025年の抽出
    path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(path, 'rb') as f: results = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    # 2025年抽出
    df_target = results[results['race_id'].astype(str).str.startswith('2025')].copy()
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    del results
    gc.collect()
    print(f"Target Races (2025): {df_target['race_id'].nunique()}", flush=True)

    # 2. 馬成績データ
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    gc.collect() 
    
    # 3. その他
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    # 4. モデル
    model = HorseRaceModel()
    model.load(MODEL_PATH)
    
    try:
        with open(os.path.join(MODEL_BASE_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_BASE_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except:
        print("Using default processor/engineer", flush=True)
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)
    return predictor, hr, peds, df_target

def process_batch_safe(df_batch, predictor, hr, peds):
    """データリークを防止したバッチ予測"""
    try:
        df = df_batch.copy()
        df.columns = df.columns.str.replace(' ', '')
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        df_proc = predictor.processor.process_results(df)
        df_proc = predictor.engineer.add_horse_history_features(df_proc, hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)

        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            if c in df_proc.columns:
                X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
            else:
                X[c] = 0
            
        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
    except Exception:
        traceback.print_exc()
        return None

def calculate_confidence_details(race_df):
    """自信度とその要因を返す"""
    if race_df.empty: return 'D', 0, 0
    
    df_sorted = race_df.sort_values('probability', ascending=False)
    top = df_sorted.iloc[0]
    
    top_prob = top['probability']
    top_ev = top['expected_value']
    
    if top_prob >= 0.5 or top_ev >= 1.5: conf = 'S'
    elif top_prob >= 0.4 or top_ev >= 1.2: conf = 'A'
    elif top_prob >= 0.3 or top_ev >= 1.0: conf = 'B'
    elif top_prob >= 0.2: conf = 'C'
    else: conf = 'D'
    
    return conf, top_prob, top_ev

def analyze():
    predictor, hr, peds, df_target = load_resources()
    
    race_ids = sorted(df_target['race_id'].unique())
    race_chunks = [race_ids[i:i + BATCH_SIZE] for i in range(0, len(race_ids), BATCH_SIZE)]
    
    data = []
    print(f"Processing {len(race_chunks)} batches...", flush=True)
    
    for chunk in tqdm(race_chunks):
        df_chunk = df_target[df_target['race_id'].isin(chunk)].copy()
        df_preds = process_batch_safe(df_chunk, predictor, hr, peds)
        
        if df_preds is None: continue
        
        for race_id, race_df in df_preds.groupby('race_id'):
            if len(race_df) < 5: continue
            conf, prob, ev = calculate_confidence_details(race_df)
            data.append({
                'race_id': race_id,
                'confidence': conf,
                'top_prob': prob,
                'top_ev': ev
            })

    if not data:
        print("No data collected!")
        return

    df_res = pd.DataFrame(data)
    
    print("\n=== Confidence Distribution Analysis ===")
    print(df_res['confidence'].value_counts().sort_index())
    
    print("\n=== Statistics per Confidence ===")
    print(df_res.groupby('confidence')[['top_prob', 'top_ev']].describe())
    
    # Aの詳細
    print("\n=== Confidence A Details (Sample) ===")
    print(df_res[df_res['confidence'] == 'A'].head(10))
    
    # Sの境界付近
    print("\n=== Confidence S Details (Near Boundary) ===")
    print(df_res[(df_res['confidence'] == 'S') & (df_res['top_prob'] < 0.6) & (df_res['top_ev'] < 2.0)].head(10))

if __name__ == "__main__":
    analyze()
