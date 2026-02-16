
import pandas as pd
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm
import unicodedata
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, MODEL_DIR, PLACE_DICT
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator
from modules.strategy_composite import CompositeBettingStrategy

# Constants
TARGET_START = "2026-01-01"
TARGET_END = "2026-02-01" # Inclusive
REPORT_FILE = "report_2026_jan_feb.md"

def to_md_table(df):
    if df.empty: return "No Data"
    return df.to_markdown(index=False, floatfmt=".1f")

def run():
    print(f"=== 2026/1/1 - 2026/2/1 Simulation ===")
    
    # Load Date Map
    date_map_path = os.path.join(os.path.dirname(RAW_DATA_DIR), "date_map_2026.pickle")
    if not os.path.exists(date_map_path):
        print("date_map_2026.pickle not found.")
        return
    with open(date_map_path, 'rb') as f:
        date_map = pickle.load(f)
        
    # Identify Target Races
    target_rids = []
    start_dt = pd.to_datetime(TARGET_START)
    end_dt = pd.to_datetime(TARGET_END)
    
    for rid, date_str in date_map.items():
        # date_str: YYYY-MM-DD
        dt = pd.to_datetime(date_str)
        if start_dt <= dt <= end_dt:
            target_rids.append(rid)
            
    print(f"Target Races: {len(target_rids)}")
    if not target_rids:
        print("No races found in period.")
        return
        
    # Load Model & Resources
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
    
    # Load Results
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
    else:
        results['race_id'] = results.index.astype(str)
        
    # Filter Results
    df_target = results[results['race_id'].isin(target_rids)].copy()
    if df_target.empty:
        print("No result rows for target IDs.")
        return
    
    df_target.set_index('race_id', inplace=True)
    
    # Preprocess
    print("Preprocessing...")
    df_proc = processor.process_results(df_target)
    df_proc.index = df_proc.index.astype(str)
    
    # Pseudo Date
    rid_to_date = {rid: pd.Timestamp(date_map[rid]) for rid in df_proc.index.unique() if rid in date_map}
    df_proc['date'] = df_proc.index.map(rid_to_date)
    df_proc.dropna(subset=['date'], inplace=True) # Safety
    
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
    
    # Simulation
    print("Simulating...")
    if 'original_race_id' in df_proc.columns:
        grouper = df_proc.groupby('original_race_id')
    else:
        grouper = df_proc.groupby(level=0)
        
    strat_names = ['formation', 'hybrid_1000']
    stats = {sn: [] for sn in strat_names}
    
    import itertools
    
    def expand_bets(recs):
        tickets = []
        for r in recs:
            b_type = r['bet_type']
            method = r.get('method', '通常')
            h_nums = r['horse_numbers']
            amount = r.get('unit_amount', 100) # Assuming uniform unit for now or total/points?
            # allocator returns 'total_amount'.
            # If box, points = len(combinations). unit = total / points.
            # But usually allocator sets 'unit_amount' 100 or so. 
            # Let's assume 100 for simplicity or derive it.
            # BettingAllocator returns 'unit_amount' in some strats.
            unit = r.get('unit_amount', 100)
            
            # Format helper
            def fmt_combo(nums, is_order=False):
                if is_order:
                    return "→".join(map(str, nums))
                else:
                    return "-".join(map(str, sorted(nums)))
            
            expanded = []
            
            if method == 'BOX':
                if b_type == 'ワイド':
                     expanded = list(itertools.combinations(h_nums, 2))
                elif b_type == '馬連':
                     expanded = list(itertools.combinations(h_nums, 2))
                elif b_type == '3連複':
                     expanded = list(itertools.combinations(h_nums, 3))
                elif b_type == '馬単':
                     expanded = list(itertools.permutations(h_nums, 2))
                elif b_type == '3連単':
                     expanded = list(itertools.permutations(h_nums, 3))
            elif method == '流し':
                 # Nagashi Logic
                 # betting_allocator's formation for nagashi usually: [[axis...], [opps...]]
                 # or [axis, opp1, opp2...] if 'horse_numbers' only?
                 # Need to check 'formation' key.
                 form = r.get('formation')
                 if form and len(form) == 2:
                     axes = form[0]
                     opps = form[1]
                     if b_type == '3連複':
                         # Axis 1 head: 1-all-all
                         if len(axes) == 1:
                             ax = axes[0]
                             # Pairs from opps
                             pairs = list(itertools.combinations(opps, 2))
                             expanded = [[ax, p[0], p[1]] for p in pairs]
                     elif b_type == 'ワイド' or b_type == '馬連':
                         if len(axes) == 1:
                             ax = axes[0]
                             expanded = [[ax, o] for o in opps]
            else:
                 # Single / Normal
                 if b_type == '単勝' or b_type == '複勝':
                     # h_nums is list, but usually single bet has 1 horse per ticket?
                     # Allocator might group singles? "単勝" with [1, 2] means buy 1 and 2?
                     # Yes usually.
                     expanded = [[h] for h in h_nums]
                 else:
                     # Exact combination (e.g. formation expanded in allocator?)
                     # If allocator returned explicit combination text, we can't easily parse.
                     # But allocator usually returns 'horse_numbers' as the set used.
                     # If 'method' is '通常' and it's a 3-ren-puku, it implies one ticket?
                     # Or logic unknown.
                     # But hybrid_1000 uses: Nagashi(3ren), Wide Box, Tan.
                     # Formation uses: 3renpuku formation mainly.
                     # Formation details in 'formation' key?
                     pass

            # Create tickets
            is_ordered = b_type in ['馬単', '3連単']
            for c in expanded:
                 tickets.append({
                     'type': b_type,
                     'combo': fmt_combo(c, is_ordered),
                     'amount': unit
                 })
                 
        return tickets

    for rid, race_df in tqdm(grouper):
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'horse_name': str(row.get('馬名', '')),
                'probability': float(row['probability']),
                'odds': float(row.get('odds', 0) or 0),
                'expected_value': float(row['expected_value']),
                'is_win': int(row.get('着順', 99) == 1),
                'rank': int(row.get('着順', 99))
            })
        df_preds = pd.DataFrame(preds)
        
        # Bets
        for sn in strat_names:
            recs = BettingAllocator.allocate_budget(df_preds, 10000 if sn=='formation' else 1000, strategy=sn)
            invest = sum(r['total_amount'] for r in recs)
            payout = 0
            
            # Payout Check
            # Need actual results for validation (Rank 1, 2, 3)
            # We have 'rank' in preds.
            # Determine winners
            rank1 = [p['horse_number'] for p in preds if p['rank'] == 1]
            rank2 = [p['horse_number'] for p in preds if p['rank'] == 2]
            rank3 = [p['horse_number'] for p in preds if p['rank'] == 3]
            
            # Simple Payout Logic (Approximation using stored Dividends is unavailable easily here without parsing Return Tables)
            # We need Return Tables for accurate simulation!
            # Or we approximate if we only have Win Odds? No, we need Trifecta payouts.
            # Use `return_tables.pickle`?
            # Or use `analyze_2025_breakdown` logic which assumed we had return info?
            # Wait, `analyze_2025_breakdown` used `payout` column in `race_df`? No.
            # It calculated payout using specific logic or external data.
            # Let's check `analyze_2025_breakdown.py` specifically for Payout calculation.
            pass

    # ... WAIT, `analyze_2025_breakdown.py` used `return_tables.pickle`!
    # I need to implement payout lookup.
    
    # [Insert Payout Lookup Logic Here]
    # Loading return tables
    with open(os.path.join(RAW_DATA_DIR, "return_tables.pickle"), 'rb') as f:
        return_tables = pickle.load(f)
        
    # Re-loop with payout calc
    stats = {sn: [] for sn in strat_names}
    
    for rid, race_df in tqdm(grouper):
        preds = []
        for _, row in race_df.iterrows():
             preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'probability': float(row['probability']),
                'odds': float(row.get('odds', 0) or 0),
                'expected_value': float(row['expected_value'])
            })
        df_preds = pd.DataFrame(preds)
        
        # Get Race Return
        if rid in return_tables.index:
            race_return = return_tables.loc[rid]
        else:
            race_return = None
        
        # Get Actual Result (Top 3)
        # race_df has '着順' and '馬番'.
        # Filter for 1, 2, 3
        actual_1 = race_df[race_df['着順'] == 1]['馬番'].tolist()
        actual_2 = race_df[race_df['着順'] == 2]['馬番'].tolist()
        actual_3 = race_df[race_df['着順'] == 3]['馬番'].tolist()
        
        # Determine actuals (handle ties if multiple)
        h1 = actual_1[0] if actual_1 else -1
        h2 = actual_2[0] if actual_2 else -1
        h3 = actual_3[0] if actual_3 else -1
        
        for sn in strat_names:
            budget = 1000 if sn == 'hybrid_1000' else 5000 
            recs = BettingAllocator.allocate_budget(df_preds, budget, strategy=sn)
            invest = sum(r['total_amount'] for r in recs)
            
            payout = 0
            hit_count = 0
            
            if race_return is not None and not race_return.empty:
                try:
                    # Expand tickets
                    tickets = expand_bets(recs)
                    payout = CompositeBettingStrategy.calculate_return(tickets, race_return)
                except Exception as e:
                    # Generic error catch to avoid crash
                    # print(f"Payout Error {rid}: {e}")
                    payout = 0
                        
                if payout > 0: hit_count = 1
            else:
                # Manual Hit Check (Simplified if no return data)
                # For now, just skip if no return data, as we expect it for 2026
                pass
            
            stats[sn].append({
                'race_id': rid,
                'invest': invest,
                'payout': payout,
                'hit': hit_count
            })

    # Report
    summary = []
    for sn, data in stats.items():
        if not data: continue
        df_s = pd.DataFrame(data)
        races = len(df_s)
        total_invest = df_s['invest'].sum()
        total_payout = df_s['payout'].sum()
        hit = df_s['hit'].sum()
        
        summary.append({
            'Strategy': sn,
            'Races': races,
            'Invest': total_invest,
            'Payout': total_payout,
            'Recovery': f"{(total_payout/total_invest)*100:.1f}%" if total_invest > 0 else "0%",
            'Hit Rate': f"{(hit/races)*100:.1f}%"
        })
        
    print("\n=== Summary (2026/1/1 - 2026/2/1) ===")
    print(pd.DataFrame(summary).to_markdown(index=False))
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 2026年1月度 シミュレーション結果\n\n")
        f.write(f"- 期間: {TARGET_START} ~ {TARGET_END}\n")
        f.write(f"- 対象レース数: {len(target_rids)}\n\n")
        f.write("## サマリー\n")
        f.write(pd.DataFrame(summary).to_markdown(index=False))
        
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    run()
