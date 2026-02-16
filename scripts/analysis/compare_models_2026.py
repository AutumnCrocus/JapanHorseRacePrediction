
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, RESULTS_FILE, RETURN_FILE, HORSE_RESULTS_FILE, PEDS_FILE, MODEL_DIR
from modules.training import HorseRaceModel
from modules.betting_allocator import BettingAllocator
from modules.strategy_composite import CompositeBettingStrategy
import itertools

# Comparison Config
PERIOD_START = '2026-01-01'
PERIOD_END = '2026-02-01'
STRATEGY = 'hybrid_1000'
BUDGET = 1000

def load_model_env(model_dir):
    print(f"Loading model from {model_dir}...")
    model = HorseRaceModel()
    model.load(os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f: proc = pickle.load(f)
    with open(os.path.join(model_dir, 'engineer.pkl'), 'rb') as f: eng = pickle.load(f)
    with open(os.path.join(model_dir, 'bias_map.pkl'), 'rb') as f: bias = pickle.load(f)
    with open(os.path.join(model_dir, 'jockey_stats.pkl'), 'rb') as f: jock = pickle.load(f)
    return model, proc, eng, bias, jock

def expand_bets(recs):
    # Same helper as in simulate_2026_period.py
    tickets = []
    for r in recs:
        b_type = r['bet_type']
        method = r.get('method', '通常')
        h_nums = r['horse_numbers']
        unit = r.get('unit_amount', 100)
        
        def fmt_combo(nums, is_order=False):
            if is_order: return "→".join(map(str, nums))
            else: return "-".join(map(str, sorted(nums)))
        
        expanded = []
        import itertools
        
        if method == 'BOX':
            if b_type == 'ワイド': expanded = list(itertools.combinations(h_nums, 2))
            elif b_type == '馬連': expanded = list(itertools.combinations(h_nums, 2))
            elif b_type == '3連複': expanded = list(itertools.combinations(h_nums, 3))
            elif b_type == '馬単': expanded = list(itertools.permutations(h_nums, 2))
            elif b_type == '3連単': expanded = list(itertools.permutations(h_nums, 3))
        elif method == '流し':
             form = r.get('formation')
             if form and len(form) == 2:
                 axes = form[0]
                 opps = form[1]
                 if b_type == '3連複':
                     if len(axes) == 1:
                         ax = axes[0]
                         pairs = list(itertools.combinations(opps, 2))
                         expanded = [[ax, p[0], p[1]] for p in pairs]
                 elif b_type == 'ワイド' or b_type == '馬連':
                     if len(axes) == 1:
                         ax = axes[0]
                         expanded = [[ax, o] for o in opps]
        else:
             if b_type == '単勝' or b_type == '複勝':
                 expanded = [[h] for h in h_nums]
        
        is_ordered = b_type in ['馬単', '3連単']
        for c in expanded:
             tickets.append({'type': b_type, 'combo': fmt_combo(c, is_ordered), 'amount': unit})
             
    return tickets

def run_sim(df_target, model, processor, engineer, bias_map, jockey_stats, hr, peds, return_tables):
    # Feature Engineering
    df_proc = processor.process_results(df_target.copy())
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc = engineer.add_jockey_features(df_proc, jockey_stats)[0]
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    df_proc = engineer.add_bias_features(df_proc, bias_map)
    df_proc = processor.encode_categorical(df_proc, ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam'])
    
    # Predict
    if df_proc.empty:
        print("Warning: df_proc is empty after feature engineering.")
        return {'Races': 0, 'Invest': 0, 'Payout': 0, 'Hit': 0}
        
    X = df_proc[model.feature_names].fillna(0).copy()
    if X.empty:
        print("Warning: X is empty for prediction.")
        return {'Races': 0, 'Invest': 0, 'Payout': 0, 'Hit': 0}
        
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
        
    df_proc['probability'] = model.predict(X.fillna(0))
    df_proc['expected_value'] = df_proc['probability'] * df_proc.get('単勝', 0)
    
    # Sim
    if 'original_race_id' in df_proc.columns:
        grouper = df_proc.groupby('original_race_id')
    else:
        grouper = df_proc.groupby(level=0)
        
    total_invest = 0
    total_payout = 0
    total_races = 0
    total_hits = 0
    
    for rid, race_df in tqdm(grouper, leave=False):
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'probability': float(row['probability']),
                'odds': float(row.get('odds', 0) or 0),
                'expected_value': float(row['expected_value'])
            })
        df_preds = pd.DataFrame(preds)
        
        recs = BettingAllocator.allocate_budget(df_preds, BUDGET, strategy=STRATEGY)
        invest = sum(r['total_amount'] for r in recs)
        
        payout = 0
        hit = 0
        if rid in return_tables.index:
            race_return = return_tables.loc[rid]
            try:
                tickets = expand_bets(recs)
                payout = CompositeBettingStrategy.calculate_return(tickets, race_return)
                if payout > 0: hit = 1
            except: pass
            
        total_invest += invest
        total_payout += payout
        total_races += 1
        total_hits += hit
        
    return {
        'Races': total_races,
        'Invest': total_invest,
        'Payout': total_payout,
        'Hit': total_hits
    }

def main():
    print("=== Model Comparison (2026/01/01 - 2026/02/01) ===")
    
    # Load Common Data
    print("Loading raw data...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, RETURN_FILE), 'rb') as f: return_tables = pickle.load(f)
    
    # First, separate 2026 data to avoid processing everything
    if 'date' not in results.columns:
        rid_str = results.index.astype(str)
        df_2026_all = results[rid_str.str.startswith('2026')].copy()
        
        if df_2026_all.empty:
            print("Error: No 2026 data found in results.pickle.")
            return
            
        print("Reconstructing pseudo-dates for 2026 filtering...")
        # Use simple logic: Year-01-01 + rank
        rid_2026 = df_2026_all.index.astype(str)
        df_2026_all['temp_rid'] = rid_2026
        df_2026_all['date_rank'] = df_2026_all['temp_rid'].rank(method='dense')
        df_2026_all['date'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(df_2026_all['date_rank'], unit='min')
        df_2026_all.drop(columns=['temp_rid', 'date_rank'], inplace=True)
        
        df_2026 = df_2026_all[(df_2026_all['date'] >= PERIOD_START) & (df_2026_all['date'] <= PERIOD_END)].copy()
    else:
        results['date'] = pd.to_datetime(results['date'], errors='coerce')
        df_2026 = results[(results['date'] >= PERIOD_START) & (results['date'] <= PERIOD_END)].copy()
    
    print(f"Target Rows: {len(df_2026)}")
    # Reset index for processing
    df_2026.index = df_2026.index.astype(str)
    
    if df_2026.empty:
        print("Error: No data found for the specified period. Check results.pickle.")
        return
        
    print(f"Target Races: {len(df_2026.index.unique())}")
    
    # 1. Sim Production Model
    print("\n--- Production Model (2016-2023 train) ---")
    prod_env = load_model_env(os.path.join(MODEL_DIR, "validation_2024"))
    res_prod = run_sim(df_2026, *prod_env, hr, peds, return_tables)
    
    # 2. Sim Historical Model
    print("\n--- Historical Model (2010-2024 train) ---")
    hist_env = load_model_env(os.path.join(MODEL_DIR, "historical_2010_2024"))
    res_hist = run_sim(df_2026, *hist_env, hr, peds, return_tables)
    
    # Compare
    def calc_metrics(d):
        rec = (d['Payout'] / d['Invest'] * 100) if d['Invest'] > 0 else 0
        hr = (d['Hit'] / d['Races'] * 100) if d['Races'] > 0 else 0
        return rec, hr
        
    rec_p, hr_p = calc_metrics(res_prod)
    rec_h, hr_h = calc_metrics(res_hist)
    
    print("\n=== Comparison Result (hybrid_1000) ===")
    print(f"{'Model':<20} | {'Invest':<10} | {'Payout':<10} | {'Recovery':<10} | {'Hit Rate':<10}")
    print("-" * 75)
    print(f"{'Production':<20} | {res_prod['Invest']:<10} | {res_prod['Payout']:<10} | {rec_p:.1f}%      | {hr_p:.1f}%")
    print(f"{'Historical':<20} | {res_hist['Invest']:<10} | {res_hist['Payout']:<10} | {rec_h:.1f}%      | {hr_h:.1f}%")

if __name__ == "__main__":
    main()
