"""
戦略比較シミュレーション (ベクトル化/バッチ処理版)
- 高速化のためにレースごとのループを排除し、バッチ単位で特徴量生成・予測を行う
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator
from modules.track_bias import TrackBiasAnalyzer

# --- 設定 ---
STRATEGIES = ['hybrid_1000', 'wide_nagashi', 'box4_sanrenpuku', 'umaren_nagashi', 'sanrenpuku_1axis', 'sanrenpuku_2axis']
BUDGETS = [500, 600, 1000, 1500]
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')
BATCH_SIZE = 100

def load_resources_original():
    print("Loading resources (Staggered for Memory Optimization)...", flush=True)
    import gc
    
    # 1. Load Results & Filter Target (Highest Priority)
    print("Loading Results...", flush=True)
    path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(path, 'rb') as f: results = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        try:
             results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except:
             results['date'] = pd.Timestamp('2024-01-01')

    # Filter for 2024-2025
    results['year'] = pd.to_datetime(results['date'], errors='coerce').dt.year
    df_all = results[results['year'].isin([2024, 2025])].copy()
    
    # Rename columns for consistency
    rename_map = {'枠 番': '枠番', '馬 番': '馬番'}
    df_all = df_all.rename(columns=rename_map)
    
    # Sort
    if 'date' in df_all.columns: df_all = df_all.sort_values(['date', 'race_id'])
    
    # Random 3000 Races as TARGET
    all_race_ids = df_all['race_id'].unique()
    if len(all_race_ids) > 3000:
        import random
        random.seed(42)  # Same seed for comparison
        target_race_ids = sorted(random.sample(list(all_race_ids), 3000))
        print(f"Randomly selected 3000 races from {len(all_race_ids)} for target simulation.", flush=True)
    else:
        target_race_ids = sorted(list(all_race_ids))
        print(f"Using all {len(target_race_ids)} races for target simulation.", flush=True)
    
    active_horses = df_all['horse_id'].unique() if 'horse_id' in df_all.columns else []
    
    # Free up huge Results df
    del results
    gc.collect()
    print(f"All 2024-2025 Races: {df_all['race_id'].nunique()} (Memory Freed)", flush=True)

    # 2. Load HR & Filter
    print("Loading Horse History...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    
    if len(active_horses) > 0:
        print(f"Filtering HR (Original: {len(hr)})...", flush=True)
        hr = hr[hr.index.isin(active_horses)].copy()
        
    gc.collect() # Compact memory
    
    # 3. Load others
    print("Loading Peds & Returns...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)
    
    # 4. Model (Last to avoid holding it during heavy data ops)
    print("Loading Model...", flush=True)
    model = HorseRaceModel()
    try:
        model.load(MODEL_PATH)
    except: pass

    try:
        with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except FileNotFoundError:
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)
        
    return predictor, hr, peds, df_all, target_race_ids, returns

def process_batch(df_batch, predictor, hr, peds):
    """バッチデータフレームに対して一括で特徴量生成・予測を行う"""
    try:
        df = df_batch.copy()
        rename_map = {'枠 番': '枠番', '馬 番': '馬番'}
        df = df.rename(columns=rename_map)
        
        # Types
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'date' not in df.columns: df['date'] = pd.Timestamp('2025-01-01')
        else: df['date'] = pd.to_datetime(df['date'], errors='coerce').fillna(pd.Timestamp('2025-01-01'))

        # Feature Engineering (Vectorized)
        # Processor
        df_proc = predictor.processor.process_results(df)
        
        # Engineer (Heavy part)
        # Note: add_horse_history_features expects 'df' with 'date' and 'horse_id'. 
        # It merges with 'hr'. If df has many rows, merge is efficient.
        # However, we must ensure 'hr' is passed correctly.
        if 'horse_id' in df.columns:
             # Just pass full HR (filtered previously)
             race_hr = hr 
        else:
             race_hr = hr

        df_proc = predictor.engineer.add_horse_history_features(df_proc, race_hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, race_hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)

        # Predict
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            X[c] = df_proc[c] if c in df_proc.columns else 0
            
        # Categorical
        try:
            debug_info = predictor.model.debug_info()
            model_cats = debug_info.get('pandas_categorical', [])
            if len(model_cats) >= 2:
                if '枠番' in X.columns:
                    cat_type = pd.CategoricalDtype(categories=model_cats[0], ordered=False)
                    X['枠番'] = X['枠番'].astype(cat_type)
                if '馬番' in X.columns:
                    cat_type = pd.CategoricalDtype(categories=model_cats[1], ordered=False)
                    X['馬番'] = X['馬番'].astype(cat_type)
        except: pass

        for c in X.columns:
             if X[c].dtype == 'object' and not isinstance(X[c].dtype, pd.CategoricalDtype):
                 X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        
        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        
        if '単勝' in df_res.columns:
            df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
        else:
            df_res['odds'] = 10.0
        
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
        
    except Exception as e:
        print(f"Batch Process Error: {e}")
        traceback.print_exc()
        return None

def verify_hit_logic(race_id, rec, returns_df):
    if race_id not in returns_df.index: return 0
    race_rets = returns_df.loc[race_id]
    if isinstance(race_rets, pd.Series): race_rets = pd.DataFrame([race_rets])
    
    payout = 0
    bet_type = rec['bet_type']
    method = rec.get('method', 'SINGLE')
    formation = rec.get('formation', []) # List of lists [[axis], [opps]] etc.
    
    hits = race_rets[race_rets[0] == bet_type]
    for _, h in hits.iterrows():
        try:
            money = int(str(h[2]).replace(',','').replace('円',''))
            win_str = str(h[1]).replace('→','-')
            if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
            else: win_nums = [int(win_str)]
            
            is_hit = False
            
            if method == 'BOX':
                # Box: Any subset of pool matches
                pool = set(rec['horse_numbers'])
                if set(win_nums).issubset(pool):
                    is_hit = True
            elif method in ['流し', '2軸流し']:
                # Nagashi/2-Axis: All axes must match, others from opponents
                # Nagashi: formation = [[axis], [opponents]]
                # 2-Axis: formation = [[axis1, axis2], [opponents]]
                axes = set(formation[0])
                opponents = set(formation[1])
                
                win_set = set(win_nums)
                if axes.issubset(win_set):
                    remaining_winners = win_set - axes
                    if remaining_winners.issubset(opponents):
                        is_hit = True
            elif method == 'FORMATION':
                # Formation: One from each group must match
                # formation = [[G1], [G2], [G3]] or [[G1], [G2]]
                if len(formation) == 3: # 3-Ren-Tan/Puku
                    # Try all permutations for 3-Ren-Puku if it's Puku
                    if bet_type == '3連複':
                         import itertools
                         for p in itertools.permutations(win_nums):
                             if p[0] in formation[0] and p[1] in formation[1] and p[2] in formation[2]:
                                 is_hit = True; break
                    else: # 3-Ren-Tan (Ordered)
                         if win_nums[0] in formation[0] and win_nums[1] in formation[1] and win_nums[2] in formation[2]:
                             is_hit = True
                elif len(formation) == 2: # Uma-Ren/Wide
                    if bet_type == 'ワイド':
                         # Any 2 from [G1, G2]
                         # Wide Formation: usually [[axis], [opp]]
                         if win_nums[0] in formation[0] and win_nums[1] in formation[1]: is_hit = True
                         elif win_nums[1] in formation[0] and win_nums[0] in formation[1]: is_hit = True
                    else: # Uma-Ren/Tan
                         if win_nums[0] in formation[0] and win_nums[1] in formation[1]: is_hit = True
                         if bet_type == '馬連' and win_nums[1] in formation[0] and win_nums[0] in formation[1]: is_hit = True
            elif method == 'SINGLE' or bet_type == '単勝':
                if win_nums[0] in rec['horse_numbers']:
                    is_hit = True
            
            if is_hit:
                # Add payout (scale by unit_amount)
                # verify_hit_logic is per ticket. scale by unit_amount/100
                payout += (money * rec.get('unit_amount', 100) // 100)
        except Exception as e:
            # print(f"Error in verification: {e}")
            pass
    return payout

def run_vectorized_simulation():
    bias_analyzer = TrackBiasAnalyzer()
    
    print("=== トラックバイアス戦略シミュレーション ===", flush=True)
    
    predictor, hr, peds, df_all, target_race_ids, returns = load_resources_original()
    
    print(f"Target Races: {len(target_race_ids)} (Sampled from {df_all['race_id'].nunique()} total 2024-2025 races)", flush=True)
    
    total_results = {
        (strat, bud): {'cost': 0, 'return': 0, 'hits': 0, 'total': 0}
        for strat in STRATEGIES for bud in BUDGETS
    }
    
    # Pre-calculate daily data grouping for fast bias lookup
    # Add 'venue_id' if not present (from race_id)
    # race_id example: '202405010101' -> year(4) venue(2) kai(2) day(2) round(2)
    # venue is index 4-6 (0-indexed)
    df_all['venue'] = df_all['race_id'].astype(str).str[4:6]
    df_all['race_no'] = df_all['race_id'].astype(str).str[10:12].astype(int)
    
    # Split TARGET races into batches
    race_chunks = [target_race_ids[i:i + BATCH_SIZE] for i in range(0, len(target_race_ids), BATCH_SIZE)]
    
    print(f"Processing {len(race_chunks)} batches (Size={BATCH_SIZE})...", flush=True)
    
    for chunk in tqdm(race_chunks):
        # 1. Batch Prediction
        df_chunk = df_all[df_all['race_id'].isin(chunk)].copy()
        
        df_preds = process_batch(df_chunk, predictor, hr, peds)
        if df_preds is None: continue
        
        # 2. Sequential Allocation (Memory efficient)
        # Group by race_id
        grouped = df_preds.groupby('race_id')
        
        for race_id, race_df_preds in grouped:
            if len(race_df_preds) < 6: continue
            
            # Extract Race Info
            first_row = race_df_preds.iloc[0]
            r_date = first_row.get('date')
            r_venue = str(race_id)[4:6]
            r_no = int(str(race_id)[10:12])
            
            # Analyze Bias (Only if needed)
            bias_info = None
            if 'track_bias' in STRATEGIES:
                day_venue_races = df_all[
                    (df_all['date'] == r_date) & 
                    (df_all['venue'] == r_venue)
                ]
                bias_info = bias_analyzer.analyze_bias(r_date, r_venue, r_no, day_venue_races)
                
                # Debug bias info occasionally
                if total_results[(STRATEGIES[0], BUDGETS[0])]['total'] % 50 == 0:
                     print(f"DEBUG BIAS {race_id}: {bias_info}", flush=True)
            
            for strat in STRATEGIES:
                for bud in BUDGETS:
                    key = (strat, bud)
                    try:
                        # Construct odds_data from results (Confirmed Odds as Voting Odds)
                        # Note: Simulation only has Win odds readily available in df_preds['odds']
                        # We construct a simple dictionary for 'tan' (Win) odds.
                        # Other bet types (Wide, Trio) heavily depend on raw odds data which is partial or missing here.
                        # However, providing 'tan' odds allows Odds Divergence / Kelly to function for Win bets.
                        odds_data = {'tan': {}}
                        for _, row in race_df_preds.iterrows():
                            try:
                                h_num = int(row['馬番'])
                                odd = float(row['odds'])
                                if odd > 0:
                                    odds_data['tan'][h_num] = odd
                            except: pass
                            
                        recs = BettingAllocator.allocate_budget(race_df_preds, bud, strategy=strat, odds_data=odds_data, bias_info=bias_info)
                        if recs:
                            for rec in recs:
                                cost = rec.get('total_amount', rec.get('amount', 0))
                                total_results[key]['cost'] += cost
                                
                                # DEBUG: Print sample bets
                                if total_results[key]['cost'] < 5000: # Print first few bets
                                    print(f"DEBUG BET: {rec['bet_type']} {rec.get('combo_str','?')} Odds:{rec.get('odds',0):.1f} Prob:{rec.get('prob',0):.3f} Div:{rec.get('divergence',0):.2f}")
                                
                                pay = verify_hit_logic(race_id, rec, returns)
                                total_results[key]['return'] += pay
                                if pay > 0:
                                     print(f"DEBUG HIT!: {rec['bet_type']} {rec.get('combo_str','?')} Pay:{pay}")
                                     total_results[key]['hits'] += 1
                                if cost > 0: total_results[key]['total'] += 1 # Only count races we bet on? Or per bet ticket?
                                # Consistent with previous logic: 'total' accumulates per ticket usually but here per race call if >0
                                # Previous: total += 1 per ticket. Let's match.
                    except Exception as e:
                        # print(e) # Suppress for cleaner output during simulation
                        pass
                    
    # Report
    report_path = 'simulation_report_sanrenpuku_nagashi.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 3連複流し戦略シミュレーションレポート\n\n")
        f.write(f"- 期間: 2024-2025 (ランダム3000レース)\n\n")
        f.write("| 戦略 | 予算 | 投資額 | 回収額 | 回収率 | 的中数 | 投資点数 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        print("\n=== 結果 ===")
        print(f"{'戦略':<10} {'予算':<6} {'投資額':<10} {'回収額':<10} {'回収率'}")
        
        for strat in STRATEGIES:
            for bud in BUDGETS:
                r = total_results[(strat, bud)]
                recov = (r['return']/r['cost']*100) if r['cost']>0 else 0
                f.write(f"| {strat} | {bud} | {r['cost']} | {r['return']} | {recov:.1f}% | {r['hits']} | {r['total']} |\n")
                print(f"{strat:<10} {bud:<6} {r['cost']:<10} {r['return']:<10} {recov:.1f}%")
                
    print(f"Report saved: {report_path}")

if __name__ == "__main__":
    run_vectorized_simulation()
