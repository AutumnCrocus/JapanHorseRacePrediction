"""
戦略比較シミュレーション
- 戦略: balance, formation (2通り)
- 予算: 1000, 5000, 10000円 (3通り)
- 学習データ: 2016-2023年
- テストデータ: 2024-2025年全レース
"""
import os
import sys

# Set threading limits to prevent import hangs
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# --- 設定 ---
STRATEGIES = ['balance', 'formation']
BUDGETS = [500, 1000, 5000, 10000]
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')

def load_resources():
    print("Loading resources (Optimized for Memory)...", flush=True)
    
    # 1. First Load Results & Filter (to identify needed horses)
    print("Loading results...", flush=True)
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(results_path, 'rb') as f: results = pickle.load(f)
    print("Results loaded.", flush=True)
    
    # Fix RaceID & Date early
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
    
    if 'date' not in results.columns:
        try: results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except: results['date'] = pd.Timestamp('2024-01-01')

    # Filter for 2025 (Test Data)
    results['year'] = pd.to_datetime(results['date'], errors='coerce').dt.year
    df_target = results[results['year'] == 2025].copy()
    
    # Sort by date
    if 'date' in df_target.columns:
        df_target = df_target.sort_values('date')
        
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    print(f"Target Races (2025): {len(df_target['race_id'].unique())}, Active Horses: {len(active_horses)}", flush=True)
    
    # Release full results
    del results
    import gc
    gc.collect()

    # 2. Load HR & Filter
    print("Loading & Filtering horse_results...", flush=True)
    try:
        with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
        if len(active_horses) > 0:
            hr = hr[hr.index.isin(active_horses)].copy()
        gc.collect()
        print("HR processed.", flush=True)
    except FileNotFoundError as e:
        print(f"CRITICAL: HR file missing: {e}", flush=True)
        raise e

    # 3. Load Peds & Returns
    print("Loading peds & returns...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)
    print("Peds/Returns loaded.", flush=True)

    # 4. Load Model (Last to keep memory free for data loading)
    print("Loading model...", flush=True)
    model = HorseRaceModel()
    try:
        model.load(MODEL_PATH)
    except:
        pass # Handle if needed

    try:
        with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except:
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)
    
    return predictor, hr, peds, df_target, returns

def process_race_data(race_row_df, predictor, hr, peds):
    """
    1レース分のRawデータ(results.pickleの行)を受け取り、モデル入力用のDataFrameを作成する
    """
    df = race_row_df.copy()
    
    # マッピング: Raw -> Processor期待名
    rename_map = {
        '枠 番': '枠番',
        '馬 番': '馬番',
    }
    df = df.rename(columns=rename_map)
    
    # 日付処理
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce').fillna(pd.Timestamp('2025-01-01'))
    else:
        df['date'] = pd.Timestamp('2025-01-01')

    # 型変換
    for col in ['馬番', '枠番', '単勝']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    try:
        # 1. 前処理 (DataProcessor)
        df_proc = predictor.processor.process_results(df)
        
        # 2. 特徴量エンジニアリング (関連データ結合)
        # HR (馬過去成績)
        if 'horse_id' in df.columns:
            race_hr = hr[hr.index.isin(df['horse_id'].unique())]
        else:
            race_hr = hr # 全量使用(遅いかも)
        
        df_proc = predictor.engineer.add_horse_history_features(df_proc, race_hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, race_hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc) # Jockey stats might need mock if not persistent
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)
        
        # 3. 特徴量選択
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            X[c] = df_proc[c] if c in df_proc.columns else 0
            
        # 4. カテゴリカル対応 (LightGBM用)
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
        except:
            pass

        # 5. Object型を数値へ
        for c in X.columns:
             if X[c].dtype == 'object' and not isinstance(X[c].dtype, pd.CategoricalDtype):
                 X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

        # 6. 予測
        probs = predictor.model.predict(X)
        
        # 結果整形
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
        return None

def verify_hit(race_id, rec, returns_df):
    """的中判定と払戻金計算"""
    if race_id not in returns_df.index: return 0
    race_rets = returns_df.loc[race_id]
    
    payout = 0
    bet_type = rec['bet_type']
    method = rec.get('method', 'SINGLE')
    
    # 該当券種の払戻レコード抽出
    if isinstance(race_rets, pd.Series):
        hits = pd.DataFrame([race_rets])
    else:
        hits = race_rets
    
    hits = hits[hits[0] == bet_type]
    
    for _, h in hits.iterrows():
        try:
            money_str = str(h[2]).replace(',', '').replace('円', '')
            pay = int(money_str)
            win_str = str(h[1]).replace('→', '-')
            
            if '-' in win_str:
                win_nums = [int(x) for x in win_str.split('-')]
            else:
                win_nums = [int(win_str)]
                
            is_hit = False
            
            # --- Strict Hit Verification ---
            
            # 1. Box Logic
            if method == 'BOX':
                 # Box: All winning numbers must be in the selected horses
                 bet_horse_nums = set(rec['horse_numbers'])
                 if set(win_nums).issubset(bet_horse_nums):
                     is_hit = True
                     
            # 2. Formation Logic
            elif method in ['FORMATION', '流し']:
                 # Formation should have 'formation' list: [g1, g2, g3]
                 structure = rec.get('formation', [])
                 
                 if not structure:
                     # Fallback if structure missing (should not happen in corrected allocator)
                     if set(win_nums).issubset(set(rec['horse_numbers'])):
                         is_hit = True 
                 else:
                     # Check based on bet type length
                     if bet_type == '3連単':
                         if len(win_nums) == 3 and len(structure) >= 3:
                             # 1st in g1, 2nd in g2, 3rd in g3
                             if win_nums[0] in structure[0] and \
                                win_nums[1] in structure[1] and \
                                win_nums[2] in structure[2]:
                                 is_hit = True
                     elif bet_type == '3連複':
                         if len(win_nums) == 3:
                             # 3-Ren-Puku Formation usually: 1-Axis (g1 -> g2)
                             # Means: One horse from g1 must be in winners, others from g2
                             # OR more complex. Assuming standard Allocator structure:
                             # If structure has 2 groups [g1, g2]:
                             #  g1 is axis (must be in winners)
                             #  g2 is opponent (rest of winners must be in g2)
                             # BUT Allocator might put ALL horses in g2 (including g1) or not.
                             # Let's check overlap count.
                             
                             if len(structure) == 2:
                                 g1 = set(structure[0])
                                 g2 = set(structure[1])
                                 winners = set(win_nums)
                                 
                                 # Axis hit?
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1: # Assuming 1-head axis
                                     rem_winners = winners - axis_hit
                                     if rem_winners.issubset(g2):
                                         is_hit = True
                             elif len(structure) == 1: # Box
                                 if set(win_nums).issubset(set(structure[0])):
                                     is_hit = True
                                     
                     elif bet_type in ['馬連', 'ワイド']:
                         req_len = 2
                         if len(win_nums) == 2:
                             if len(structure) == 2: # Nagashi
                                 g1 = set(structure[0])
                                 g2 = set(structure[1])
                                 winners = set(win_nums)
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1:
                                     rem_winners = winners - axis_hit
                                     if rem_winners.issubset(g2):
                                         is_hit = True
                             elif len(structure) == 1: # Box
                                 if set(win_nums).issubset(set(structure[0])):
                                     is_hit = True

            # 3. Single Logic
            elif method == 'SINGLE':
                bet_horse_nums = rec['horse_numbers']
                if bet_type == '単勝':
                    if win_nums[0] in bet_horse_nums: is_hit = True
                else:
                    # Regular single bet (Exact match of combination)
                    # For simple comparison, sort if order doesn't matter?
                    # But Single usually implies specific ticket.
                    # Allocator currently doesn't output loose singles except Tan/Fuku.
                    if list(win_nums) == list(bet_horse_nums):
                         is_hit = True

            if is_hit:
                # 100円あたりの払戻 * (購入額/100)
                # Note: 'total_amount' is total cost. 'unit_amount' is price per point.
                unit = rec.get('unit_amount', 100) 
                payout += int(pay * (unit / 100))
                
        except Exception as e:
            # print(f"Hit check error: {e}")
            continue
            
    return payout

def run_simulation():
    print("=== 戦略比較シミュレーション (Optimized) ===")
    
    predictor, hr, peds, df_target, returns = load_resources()
    
    try:
        race_ids = df_target['race_id'].unique()
    except:
        print("Error: Could not retrieve race_ids from loaded target df.")
        return
    
    # 結果格納
    results = {
        (strat, bud): {'cost': 0, 'return': 0, 'hits': 0, 'total': 0}
        for strat in STRATEGIES for bud in BUDGETS
    }
    
    print("Running simulation...")
    for race_id in tqdm(race_ids, mininterval=1.0):
        race_rows = df_target[df_target['race_id'] == race_id]
        if len(race_rows) < 6: continue
        
        # Feature Engineering + Prediction
        df_preds = process_race_data(race_rows, predictor, hr, peds)
        if df_preds is None or df_preds.empty:
            continue
            
        for strat in STRATEGIES:
            for bud in BUDGETS:
                try:
                    recs = BettingAllocator.allocate_budget(df_preds, bud, strategy=strat)
                except:
                    continue
                    
                if not recs: continue
                
                start_cost = results[(strat, bud)]['cost']
                start_ret = results[(strat, bud)]['return']
                
                for rec in recs:
                    cost = rec.get('total_amount', rec.get('amount', 0))
                    results[(strat, bud)]['cost'] += cost
                    results[(strat, bud)]['total'] += 1
                    
                    pay = verify_hit(race_id, rec, returns)
                    results[(strat, bud)]['return'] += pay
                    if pay > 0:
                        results[(strat, bud)]['hits'] += 1
                        
    # Report
    report_path = 'simulation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 戦略比較シミュレーションレポート\n\n")
        f.write(f"- 期間: 2024-2025 ({len(race_ids)}レース)\n\n")
        f.write("| 戦略 | 予算 | 投資額 | 回収額 | 回収率 | 的中数 | 総点数 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        print("\n=== 結果 ===")
        print(f"{'戦略':<10} {'予算':<6} {'投資額':<10} {'回収額':<10} {'回収率'}")
        
        for strat in STRATEGIES:
            for bud in BUDGETS:
                r = results[(strat, bud)]
                recov = (r['return']/r['cost']*100) if r['cost']>0 else 0
                f.write(f"| {strat} | {bud} | {r['cost']} | {r['return']} | {recov:.1f}% | {r['hits']} | {r['total']} |\n")
                print(f"{strat:<10} {bud:<6} {r['cost']:<10} {r['return']:<10} {recov:.1f}%")

    print(f"Report saved: {report_path}")

if __name__ == "__main__":
    run_simulation()
