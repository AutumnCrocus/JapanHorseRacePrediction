
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator
from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE

print(f"BettingAllocator loaded from: {BettingAllocator.__module__}")

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
RACE_ID = '202503030611'

def load_resources():
    print("Loading data...")
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
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    
    df_target = results[results['race_id'].astype(str) == RACE_ID].copy()
    
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    
    return df_target, hr, peds

def load_predictor():
    print("Loading model...")
    model = HorseRaceModel()
    model.load(os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl'))
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    return RacePredictor(model, processor, engineer)

def process_race(race_df, predictor, hr, peds):
    df = race_df.copy()
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
    df_res['horse_number'] = df_res['馬番']
    df_res['odds'] = pd.to_numeric(df_res.get('単勝', 10), errors='coerce').fillna(10.0)
    
    return df_res

def main():
    df, hr, peds = load_resources()
    if df.empty:
        print(f"Race {RACE_ID} not found.")
        return
        
    predictor = load_predictor()
    df_pred = process_race(df, predictor, hr, peds)
    
    # Analyze Score logic
    top_horse = df_pred.sort_values('probability', ascending=False).iloc[0]
    top1_prob = top_horse['probability']
    top1_odds = top_horse['odds']
    
    sorted_probs = df_pred['probability'].sort_values(ascending=False)
    prob_gap = sorted_probs.iloc[0] - sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0
    
    score = 0
    if top1_odds < 5: score += 30
    elif top1_odds < 10: score += 20
    elif top1_odds < 15: score += 10
    
    if top1_prob > 0.9: score += 30
    elif top1_prob > 0.85: score += 20
    elif top1_prob > 0.7: score += 10
    
    if prob_gap > 0.7: score -= 20
    elif prob_gap > 0.5: score -= 10
    
    if 0.1 < prob_gap < 0.3: score += 20
    elif prob_gap < 0.1: score += 10
    
    print("\n" + "="*50)
    print(f"Race Analysis: {RACE_ID}")
    print("="*50)
    print(f"Top1 Prob: {top1_prob:.3f}")
    print(f"Top1 Odds: {top1_odds:.1f}")
    print(f"Prob Gap : {prob_gap:.3f}")
    print(f"Score    : {score}")
    
    print("\n--- Formation Flex (Winner) ---")
    recs_old = BettingAllocator._allocate_formation_flex(df_pred, 5000)
    for r in recs_old:
        print(r.get('desc', 'No Desc'))
        print(r.get('horses', 'No Horses'))
        
    print("\n--- Meta Contrarian (Loser) ---")
    recs_new = BettingAllocator._allocate_meta_contrarian(df_pred, 5000)
    for r in recs_new:
        print(r.get('desc', 'No Desc'))
        print(r.get('horses', 'No Horses'))

if __name__ == "__main__":
    main()
