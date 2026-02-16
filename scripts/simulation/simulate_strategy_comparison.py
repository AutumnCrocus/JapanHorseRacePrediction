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
import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE  # noqa: E402
from modules.training import HorseRaceModel, RacePredictor  # noqa: E402
from modules.betting_allocator import BettingAllocator  # noqa: E402

# --- 設定 ---
STRATEGIES = ['balance', 'formation']
BUDGETS = [500, 1000, 5000, 10000]
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')

def load_resources():
    print("Loading resources (Optimized for Memory)...", flush=True)
    
    # 1. Load Results using DataLoader
    print("Loading results (2025)...", flush=True)
    from modules.data_loader import load_results, load_payouts
    
    # Load only 2025 results for simulation
    results = load_results(2025, 2025)
    print(f"Results loaded: {len(results)} rows", flush=True)
    
    # Fix RaceID & Date early
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
    
    if 'date' not in results.columns:
        try:
            results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except Exception:
            results['date'] = pd.Timestamp('2024-01-01')

    # Filter for 2025 (Test Data)
    # DataLoader already filtered by year, but safety check
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
        with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
            hr = pickle.load(f)
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
    
    # Load payouts for 2025
    returns = load_payouts(2025, 2025)
    
    print("Peds/Returns loaded.", flush=True)

    # 4. Load Model (Last to keep memory free for data loading)
    print("Loading model...", flush=True)
    model = HorseRaceModel()
    try:
        model.load(MODEL_PATH)
    except Exception:
        pass  # Handle if needed

    try:
        with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f:
            processor = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f:
            engineer = pickle.load(f)
    except Exception:
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

# Map Japanese bet types to scrape keys
BET_TYPE_MAP = {
    '単勝': 'tan',
    '複勝': 'fuku',
    '枠連': 'wakuren',
    '馬連': 'umaren',
    'ワイド': 'wide',
    '馬単': 'umatan',
    '3連複': 'sanrenpuku',
    '3連単': 'sanrentan'
}

def verify_hit(race_id, rec, payouts_dict):
    """的中判定と払戻金計算 (Dict base)"""
    if race_id not in payouts_dict: return 0
    race_pay = payouts_dict[race_id]
    
    payout = 0
    bet_type_jp = rec['bet_type']
    bet_key = BET_TYPE_MAP.get(bet_type_jp)
    
    if not bet_key or bet_key not in race_pay:
        return 0
        
    winning_data = race_pay[bet_key] # { winning_num_or_tuple: payout_amount }
    
    # 賭けた買い目を展開して、winning_dataにあるか確認する
    # rec['horse_numbers']: main numbers
    # rec['method']: 'BOX', 'formation', 'SINGLE'
    
    method = rec.get('method', 'SINGLE')
    bet_horses = rec['horse_numbers']
    
    hits = 0
    
    import itertools
    
    # 1. 単勝・複勝 (Single number keys)
    if bet_key in ['tan', 'fuku']:
        # 通常、単複はSINGLE扱いだが、BOX等で複数買いしている場合も考慮
        for h in bet_horses:
            if h in winning_data:
                payout += winning_data[h] * (rec.get('unit_amount', 100) / 100)

    # 2. 組み合わせ馬券 (Tuple keys)
    else:
        # Generate bought combinations based on method
        bought_combinations = []
        
        if method == 'BOX':
            if bet_key in ['umaren', 'wide']:
                # 2頭組み合わせ
                bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 2)]
            elif bet_key in ['sanrenpuku']:
                # 3頭組み合わせ
                bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 3)]
        
        elif method in ['FORMATION', '流し']:
            # Formation構造 [g1, g2, g3]
            structure = rec.get('formation', [])
            if not structure:
                structure = [bet_horses]  # Fallback
            
            if bet_key in ['umaren', 'wide', 'umatan']:
                if len(structure) >= 2:
                    g1 = structure[0]
                    g2 = structure[1]
                    for h1 in g1:
                        for h2 in g2:
                            if h1 == h2: continue
                            if bet_key == 'umatan':
                                bought_combinations.append((h1, h2))
                            else:
                                bought_combinations.append(tuple(sorted((h1, h2))))
                                
            elif bet_key in ['sanrenpuku', 'sanrentan']:
                if len(structure) >= 3:
                    g1 = structure[0]
                    g2 = structure[1]
                    g3 = structure[2]
                    for h1 in g1:
                        for h2 in g2:
                            if h1 == h2: continue
                            for h3 in g3:
                                if h3 == h1 or h3 == h2: continue
                                
                                if bet_key == 'sanrentan':
                                    bought_combinations.append((h1, h2, h3))
                                else:
                                    bought_combinations.append(tuple(sorted((h1, h2, h3))))

        elif method == 'SINGLE':
            # Specific single combination (e.g. prediction top 2)
            # Assuming 'horse_numbers' is the ordered list (e.g. 1-2-3)
            # Convert to tuple
            if bet_key in ['umaren', 'wide', 'sanrenpuku']:
                bought_combinations.append(tuple(sorted(bet_horses)))
            else:
                bought_combinations.append(tuple(bet_horses))

        # Check hits
        # Deduplicate combinations (e.g. Box 1,2,3 -> (1,2), (1,3), (2,3))
        bought_combinations = set(bought_combinations)
        
        for comb in bought_combinations:
            if comb in winning_data:
                payout += winning_data[comb] * (rec.get('unit_amount', 100) / 100)
            
    return int(payout)

def run_simulation():
    print("=== 戦略比較シミュレーション (Optimized) ===")
    
    predictor, hr, peds, df_target, returns = load_resources()
    
    try:
        race_ids = df_target['race_id'].unique()
    except Exception:
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
                except Exception:
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
