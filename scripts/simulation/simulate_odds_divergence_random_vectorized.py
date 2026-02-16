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

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# --- 設定 ---
STRATEGIES = ['odds_divergence']
BUDGETS = [5000, 10000, 20000]
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')
BATCH_SIZE = 100 # バッチサイズ (レース数) - Increase for speed if memory allows

def load_resources():
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
    df_target = results[results['year'].isin([2024, 2025])].copy()
    
    # Random 500 Races
    all_race_ids = df_target['race_id'].unique()
    if len(all_race_ids) > 500:
        import random
        random.seed(42)
        selected_race_ids = sorted(random.sample(list(all_race_ids), 500))
        df_target = df_target[df_target['race_id'].isin(selected_race_ids)].copy()
        print(f"Randomly selected 500 races from {len(all_race_ids)}.", flush=True)
    
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    # Free up huge Results df
    del results
    gc.collect()
    print(f"Target Races: {df_target['race_id'].nunique()} (Memory Freed)", flush=True)

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
        
    return predictor, hr, peds, df_target, returns

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
    bet_horse_nums = set(rec['horse_numbers'])
    
    hits = race_rets[race_rets[0] == bet_type]
    for _, h in hits.iterrows():
        try:
            money = int(str(h[2]).replace(',','').replace('円',''))
            win_str = str(h[1]).replace('→','-')
            if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
            else: win_nums = [int(win_str)]
            
            is_hit = False
            if bet_type == '単勝':
                if win_nums[0] in bet_horse_nums: is_hit = True
            elif bet_type in ['馬連', 'ワイド', '3連複', '3連単']:
                if set(win_nums).issubset(bet_horse_nums): is_hit = True
                
            if is_hit:
                payout += money
        except: pass
    return payout

def run_vectorized_simulation():
    print("=== 高速シミュレーション (ベクトル化バッチ処理) ===", flush=True)
    
    predictor, hr, peds, df_target, returns = load_resources()
    
    race_ids = df_target['race_id'].unique()
    print(f"Target Races: {len(race_ids)} (2024-2025 Full)", flush=True)
    
    total_results = {
        (strat, bud): {'cost': 0, 'return': 0, 'hits': 0, 'total': 0}
        for strat in STRATEGIES for bud in BUDGETS
    }
    
    # Split races into batches
    race_chunks = [race_ids[i:i + BATCH_SIZE] for i in range(0, len(race_ids), BATCH_SIZE)]
    
    print(f"Processing {len(race_chunks)} batches (Size={BATCH_SIZE})...", flush=True)
    
    for chunk in tqdm(race_chunks):
        # 1. Batch Prediction
        df_chunk = df_target[df_target['race_id'].isin(chunk)].copy()
        
        df_preds = process_batch(df_chunk, predictor, hr, peds)
        if df_preds is None: continue
        
        # 2. Sequential Allocation (Memory efficient)
        # Group by race_id
        grouped = df_preds.groupby('race_id')
        
        for race_id, race_df_preds in grouped:
            if len(race_df_preds) < 6: continue
            
            for strat in STRATEGIES:
                for bud in BUDGETS:
                    key = (strat, bud)
                    try:
                        recs = BettingAllocator.allocate_budget(race_df_preds, bud, strategy=strat)
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
                    except: pass
                    
    # Report
    # Report
    report_path = 'simulation_report_odds_divergence_random_vectorized.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# オッズ乖離戦略シミュレーションレポート (Random 500 Vectorized)\n\n")
        f.write(f"- 期間: 2024-2025 (ランダム500レース)\n\n")
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
