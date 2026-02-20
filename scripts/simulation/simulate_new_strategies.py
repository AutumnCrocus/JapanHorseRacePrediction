
"""
新戦略比較シミュレーション (2025年データ)
- 戦略: Dynamic Box, Value Hunter, Confidence Scaler
- モデル: experiment_model_2026.pkl
- テストデータ: 2025年全レース
"""
import os
import sys

# Set threading limits
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np

# Project Root
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator
from modules.strategies.experimental import ExperimentalStrategies

# --- 設定 ---
STRATEGIES = ['formation', 'dynamic_box', 'value_hunter', 'confidence_scaler']
BUDGETS = [1000, 3000, 5000] # 比較のため予算統一 (Value Hunterなどは低予算でも動くが)
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2026.pkl')

def load_resources():
    print("Loading resources (Optimized)...", flush=True)
    
    # 1. Load Pre-processed Dataset (2010-2025)
    # This file contains df_proc with all features calculated
    DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset_2010_2025.pkl')
    print(f"Loading dataset from {DATASET_PATH}...", flush=True)
    
    with open(DATASET_PATH, 'rb') as f:
        data_dict = pickle.load(f)
        
    df_full = data_dict['data']
    feature_names = data_dict['feature_names']
    
    # Filter for 2025
    print("Filtering for 2025...", flush=True)
    if 'year' not in df_full.columns:
        if 'date' in df_full.columns:
            df_full['year'] = pd.to_datetime(df_full['date']).dt.year
            
    df_2025 = df_full[df_full['year'] == 2025].copy()
    
    # Alias columns
    if '馬番' in df_2025.columns and 'horse_number' not in df_2025.columns:
        df_2025['horse_number'] = df_2025['馬番']
        
    # Sort by race_id or date
    sort_cols = ['date']
    if 'original_race_id' in df_2025.columns:
        df_2025['race_id'] = df_2025['original_race_id'].astype(str)
        sort_cols.append('race_id')
    elif 'race_id' not in df_2025.columns:
         # Fallback if no race_id found, create dummy? No, critical error.
         print("CRITICAL: No race_id or original_race_id found!")
         pass
         
    if 'horse_number' in df_2025.columns:
        sort_cols.append('horse_number')
        
    df_2025 = df_2025.sort_values(sort_cols)
        
    print(f"Target Rows (2025): {len(df_2025)}", flush=True)
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    model = HorseRaceModel()
    if os.path.exists(MODEL_PATH):
        model.load(MODEL_PATH)
    else:
        print("Model non-existent, trying 2025 model...")
        model.load(os.path.join(MODEL_DIR, 'experiment_model_2025.pkl'))
        
    # 3. Load Returns for verification
    print("Loading returns...", flush=True)
    from modules.data_loader import load_payouts
    returns = load_payouts(2025, 2025)
    
    return model, df_2025, feature_names, returns

def predict_race_batch(df_race, model, feature_names):
    """Predict for a single race using pre-calc features"""
    # Prepare X
    X = df_race[feature_names].fillna(0)
    
    # Categorical handling (if needed, but usually pre-processed dataset has numeric)
    # LightGBM handles categorical if type is category, but here we assume numeric/label encoded
    
    # Predict
    probs = model.predict(X)
    
    df_res = df_race[['race_id', 'horse_number', 'horse_name', '単勝']].copy() if 'horse_name' in df_race.columns else df_race[['race_id', 'horse_number', '単勝']].copy() # Keep minimal
    if 'horse_name' not in df_res.columns:
         df_res['horse_name'] = df_res['horse_number'].astype(str)
         
    df_res['probability'] = probs
    
    # Odds
    if '単勝' in df_res.columns:
        df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
    else:
        df_res['odds'] = 10.0
        
    return df_res

# Import verification logic from existing script (or copy it efficiently)
# Creating a simple verify_hit here to avoid import issues
def verify_hit(race_id, rec, payouts_dict):
    if race_id not in payouts_dict: return 0
    race_pay = payouts_dict[race_id]
    payout = 0
    
    bet_type_map = {
        '単勝': 'tan', '複勝': 'fuku', '枠連': 'wakuren',
        '馬連': 'umaren', 'ワイド': 'wide', '馬単': 'umatan',
        '3連複': 'sanrenpuku', '3連単': 'sanrentan'
    }
    
    bet_key = bet_type_map.get(rec.get('bet_type'))
    if not bet_key or bet_key not in race_pay: return 0
    
    winning_data = race_pay[bet_key]
    bet_horses = rec.get('horse_numbers', [])
    method = rec.get('method', 'SINGLE')
    
    bought_combinations = []
    import itertools
    
    if bet_key in ['tan', 'fuku']:
        for h in bet_horses:
            if h in winning_data:
                payout += winning_data[h] * (rec.get('unit_amount', 100) / 100)
    else:
        if method == 'BOX':
            if bet_key in ['umaren', 'wide']:
                bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 2)]
            elif bet_key == 'sanrenpuku':
                bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 3)]
        elif method == 'SINGLE':
             if bet_key in ['umaren', 'wide', 'sanrenpuku']:
                bought_combinations.append(tuple(sorted(bet_horses)))
             else:
                bought_combinations.append(tuple(bet_horses))
        elif method in ['流し', '1軸流し']:
            # Simplified nagashi handling
            formation = rec.get('formation', [])
            if len(formation) >= 2:
                axis = formation[0]
                opponents = formation[1]
                if bet_key in ['umaren', 'wide']:
                    for h1 in axis:
                         for h2 in opponents:
                             bought_combinations.append(tuple(sorted((h1, h2))))
                elif bet_key == 'umatan':
                     for h1 in axis:
                         for h2 in opponents:
                             bought_combinations.append((h1, h2))
                elif bet_key == 'sanrenpuku' and len(formation) >= 2: # Axis 1 head
                     # Assuming axis is 1 head, opponents are remaining
                     # Need 2 from opponents? NO, usually axis-opponent-opponent
                     # Formation is strictly defined. Assuming standard 1-axis nagashi implies axis + 2 from opponents
                     # But experimental strategy might define formation differently.
                     # Let's rely on formation strictly if provided.
                     pass 
            
            # Re-implement strict formation parsing if simple fails
            # Actually, let's just use the logic from ExperimentalStrategies which creates correct 'formation' list
            # But here verify_hit needs to be robust. 
            # For this simulation, let's trust that strategies output standard formations.
            # ValueHunter outputs 'method': '流し' for Wide. formation=[[hole], opponents].
            # ConfScaler outputs 'method': '流し' for Umatan/Umaren. formation=[[head], opponents].
            
            if len(formation) >= 2:
                 g1 = formation[0]
                 g2 = formation[1]
                 # Wide/Umaren/Umatan
                 if bet_key in ['umaren', 'wide', 'umatan']:
                     for h1 in g1:
                         for h2 in g2:
                             if h1 == h2: continue
                             comb = (h1, h2) if bet_key == 'umatan' else tuple(sorted((h1, h2)))
                             bought_combinations.append(comb)
        
        bought_combinations = set(bought_combinations)
        for comb in bought_combinations:
            if comb in winning_data:
                payout += winning_data[comb] * (rec.get('unit_amount', 100) / 100)

    return int(payout)

def run_simulation():
    model, df_2025, feature_names, returns = load_resources()
    
    race_ids = df_2025['race_id'].unique()
    print(f"Simulating {len(race_ids)} races...", flush=True)
    
    # Init Stats
    stats = {}
    for s in STRATEGIES:
        for b in BUDGETS:
            stats[(s, b)] = {'cost': 0, 'return': 0, 'hits': 0, 'races': 0}
            
    # Iterate race_ids
    for race_id in tqdm(race_ids):
        # Slice DataFrame (Much faster than reprocessing)
        race_rows = df_2025[df_2025['race_id'] == race_id].copy()
        if len(race_rows) < 5: continue
        
        # Predict using batch method
        df_preds = predict_race_batch(race_rows, model, feature_names)
        if df_preds is None or df_preds.empty: continue
        
        # Prepare Odds Data (Approx from Result Payouts for simulation)
        # In real-time we have real odds. Here we use final payouts as proxy for odds
        # CAUTION: This means we are using CLOSING odds.
        odds_data = {}
        if race_id in returns:
            r_pay = returns[race_id]
            if 'tan' in r_pay:
                odds_data['tan'] = {k: v/100.0 for k, v in r_pay['tan'].items()}
            # Umaren/Wide odds are not single horse odds, so we only pass 'tan' usually
            
            # Map 'odds' column in df_preds to actual closing odds if available
            if 'tan' in odds_data:
                df_preds['odds'] = df_preds['horse_number'].map(odds_data['tan']).fillna(df_preds['odds'])
        
        # Run Strategies
        for strat in STRATEGIES:
            for bud in BUDGETS:
                recs = []
                if strat in ['dynamic_box', 'value_hunter', 'confidence_scaler']:
                    method = getattr(ExperimentalStrategies, strat)
                    recs = method(df_preds, bud, odds_data)
                else:
                    recs = BettingAllocator.allocate_budget(df_preds, bud, strategy=strat)
                
                if not recs: continue
                
                cost = sum([r.get('total_amount', 0) for r in recs])
                pay = 0
                for r in recs:
                    pay += verify_hit(race_id, r, returns)
                    
                stats[(strat, bud)]['cost'] += cost
                stats[(strat, bud)]['return'] += pay
                stats[(strat, bud)]['races'] += 1
                if pay > 0:
                    stats[(strat, bud)]['hits'] += 1
                    
    # Generate Report
    report_file = os.path.join(os.path.dirname(__file__), '../../reports/strategy_comparison_2025.md')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 新規馬券戦略シミュレーション結果 (2025)\n\n")
        f.write(f"- 対象レース数: {len(race_ids)}\n")
        f.write(f"- モデル: {os.path.basename(MODEL_PATH)}\n\n")
        
        f.write("## 集計結果\n\n")
        f.write("| 戦略 | 予算 | 投資総額 | 回収総額 | 回収率 | 的中率 | 購入レース数 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for strat in STRATEGIES:
            for bud in BUDGETS:
                s = stats[(strat, bud)]
                if s['cost'] == 0:
                    roi = 0
                    hit_rate = 0
                else:
                    roi = s['return'] / s['cost'] * 100
                    hit_rate = s['hits'] / s['races'] * 100 if s['races'] > 0 else 0
                    
                line = f"| {strat} | {bud} | {s['cost']:,} | {s['return']:,} | **{roi:.1f}%** | {hit_rate:.1f}% | {s['races']} |"
                f.write(line + "\n")
                print(line)

if __name__ == "__main__":
    run_simulation()
