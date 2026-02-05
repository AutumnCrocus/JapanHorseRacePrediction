
import pandas as pd
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm
import itertools

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, MODEL_DIR, PLACE_DICT, RETURN_FILE
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator
from modules.strategy_composite import CompositeBettingStrategy

REPORT_FILE = "report_2025_monthly.md"

def expand_bets(recs):
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

def run():
    print("=== 2025 Monthly Analysis ===")
    
    # Load Resources
    print("Loading resources...")
    VALIDATION_DIR = os.path.join(MODEL_DIR, "validation_2024")
    model = HorseRaceModel()
    model.load(os.path.join(VALIDATION_DIR, 'model.pkl'))
    with open(os.path.join(VALIDATION_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'bias_map.pkl'), 'rb') as f: bias_map = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'jockey_stats.pkl'), 'rb') as f: jockey_stats = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, "peds.pickle"), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, RETURN_FILE), 'rb') as f: return_tables = pickle.load(f)
    
    # Load Date Map
    with open(os.path.join(os.path.dirname(RAW_DATA_DIR), "date_map_2025.pickle"), 'rb') as f:
        date_map = pickle.load(f)
        
    # Load Results
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
    else:
        results['race_id'] = results.index.astype(str)
        
    # Filter 2025
    races_2025 = [rid for rid in results['race_id'].unique() if rid.startswith('2025')]
    df_target = results[results['race_id'].isin(races_2025)].copy()
    df_target.set_index('race_id', inplace=True)
    
    print(f"Target Races: {len(races_2025)}")
    
    # Preprocess
    print("Preprocessing...")
    df_proc = processor.process_results(df_target)
    df_proc.index = df_proc.index.astype(str)
    
    # Map Date
    rid_to_date = {}
    for rid in df_proc.index.unique():
        if rid in date_map:
            rid_to_date[rid] = pd.to_datetime(date_map[rid])
    df_proc['date'] = df_proc.index.map(rid_to_date)
    df_proc = df_proc.dropna(subset=['date'])
    df_proc['month'] = df_proc['date'].dt.month
    
    # Feature Engineering
    print("Feature Engineering...")
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc = engineer.add_jockey_features(df_proc, jockey_stats)[0]
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    df_proc = engineer.add_bias_features(df_proc, bias_map)
    df_proc = processor.encode_categorical(df_proc, ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam'])
    
    # Predict
    print("Predicting...")
    X = df_proc[model.feature_names].fillna(0).copy()
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
    df_proc['probability'] = model.predict(X.fillna(0))
    df_proc['expected_value'] = df_proc['probability'] * df_proc.get('単勝', 0)
    
    # Sim
    print("Simulating...")
    if 'original_race_id' in df_proc.columns:
        grouper = df_proc.groupby('original_race_id')
    else:
        grouper = df_proc.groupby(level=0)
        
    monthly_stats = {m: {'invest': 0, 'payout': 0, 'races': 0, 'hits': 0} for m in range(1, 13)}
    
    for rid, race_df in tqdm(grouper):
        month = race_df['month'].iloc[0]
        
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'probability': float(row['probability']),
                'odds': float(row.get('odds', 0) or 0),
                'expected_value': float(row['expected_value'])
            })
        df_preds = pd.DataFrame(preds)
        
        # Strategy: hybrid_1000
        recs = BettingAllocator.allocate_budget(df_preds, 1000, strategy='hybrid_1000')
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
            
        monthly_stats[month]['invest'] += invest
        monthly_stats[month]['payout'] += payout
        monthly_stats[month]['races'] += 1
        monthly_stats[month]['hits'] += hit

    # Summary
    rows = []
    total_inv = 0
    total_pay = 0
    total_races = 0
    total_hits = 0
    
    for m in range(1, 13):
        d = monthly_stats[m]
        rec = (d['payout'] / d['invest']) * 100 if d['invest'] > 0 else 0
        hr_rate = (d['hits'] / d['races']) * 100 if d['races'] > 0 else 0
        rows.append({
            'Month': f"{m}月",
            'Races': d['races'],
            'Invest': d['invest'],
            'Payout': d['payout'],
            'Recovery': f"{rec:.1f}%",
            'Hit Rate': f"{hr_rate:.1f}%"
        })
        total_inv += d['invest']
        total_pay += d['payout']
        total_races += d['races']
        total_hits += d['hits']
        
    # Total row
    tot_rec = (total_pay / total_inv) * 100 if total_inv > 0 else 0
    tot_hr = (total_hits / total_races) * 100 if total_races > 0 else 0
    rows.append({
        'Month': 'Total',
        'Races': total_races,
        'Invest': total_inv,
        'Payout': total_pay,
        'Recovery': f"{tot_rec:.1f}%",
        'Hit Rate': f"{tot_hr:.1f}%"
    })
    
    print("\n=== 2025 Monthly Result (hybrid_1000) ===")
    df_res = pd.DataFrame(rows)
    print(df_res.to_markdown(index=False))
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 2025年 月別シミュレーション結果 (hybrid_1000)\n\n")
        f.write(df_res.to_markdown(index=False))
        
    print(f"Saved to {REPORT_FILE}")

if __name__ == "__main__":
    run()
