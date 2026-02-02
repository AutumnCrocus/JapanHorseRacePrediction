"""
戦略比較シミュレーション (並列処理版)
- 戦略: balance, formation (2通り)
- 予算: 1000, 5000, 10000円 (3通り)
- 学習データ: 2016-2023年
- テストデータ: 2024-2025年全レース
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# --- 設定 ---
STRATEGIES = ['balance', 'formation']
BUDGETS = [1000, 5000, 10000]
MODEL_PATH = os.path.join(MODEL_DIR, 'production_model.pkl')

# グローバル変数 (ワーカープロセス内でのみ使用)
_predictor = None
_hr = None
_peds = None
_results_df = None
_returns_df = None

def init_worker(predictor_path, hr_path, peds_path, df_target_path, returns_path):
    """ワーカープロセスの初期化: ファイルからリソースをロード"""
    global _predictor, _hr, _peds, _results_df, _returns_df
    try:
        with open(predictor_path, 'rb') as f: _predictor = pickle.load(f)
        with open(hr_path, 'rb') as f: _hr = pickle.load(f)
        with open(peds_path, 'rb') as f: _peds = pickle.load(f)
        with open(df_target_path, 'rb') as f: _results_df = pickle.load(f)
        with open(returns_path, 'rb') as f: _returns_df = pickle.load(f)
    except Exception as e:
        print(f"Worker Init Error: {e}")
        traceback.print_exc()

def load_resources_main():
    """メインプロセスでのリソース読み込み"""
    print("Loading resources (Main Process)...")
    
    # Model & Helper objects
    model = HorseRaceModel()
    try:
        model.load(MODEL_PATH)
    except:
        pass
        
    try:
        with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except FileNotFoundError:
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)

    # Data
    print("Loading raw data...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)
    
    # Results
    path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(path, 'rb') as f: results = pickle.load(f)
    
    # Fix Data Structures
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
    
    # Filter HR for memory efficiency
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    if len(active_horses) > 0:
        print(f"Filtering Horse History: {len(hr)} -> ", end="")
        hr = hr[hr.index.isin(active_horses)].copy()
        print(f"{len(hr)} records")
        
    return predictor, hr, peds, df_target, returns

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

def process_race_job(race_id):
    """
    並列実行されるジョブ関数
    """
    try:
        # グローバルリソースを使用
        if _results_df is None: return None
        race_rows = _results_df[_results_df['race_id'] == race_id]
        if len(race_rows) < 6: return None
        
        # 1. Prediction Pipeline
        df = race_rows.copy()
        rename_map = {'枠 番': '枠番', '馬 番': '馬番'}
        df = df.rename(columns=rename_map)
        
        # Date & Type Fix
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').fillna(pd.Timestamp('2025-01-01'))
        else:
            df['date'] = pd.Timestamp('2025-01-01')
            
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Feature Engineering
        try:
            df_proc = _predictor.processor.process_results(df)
            
            if 'horse_id' in df.columns:
                race_hr = _hr[_hr.index.isin(df['horse_id'].unique())]
            else:
                race_hr = _hr
                
            df_proc = _predictor.engineer.add_horse_history_features(df_proc, race_hr)
            df_proc = _predictor.engineer.add_course_suitability_features(df_proc, race_hr)
            df_proc, _ = _predictor.engineer.add_jockey_features(df_proc)
            df_proc = _predictor.engineer.add_pedigree_features(df_proc, _peds)
            df_proc = _predictor.engineer.add_odds_features(df_proc)
            
            # Predict
            feature_names = _predictor.model.feature_names
            X = pd.DataFrame(index=df_proc.index)
            for c in feature_names:
                X[c] = df_proc[c] if c in df_proc.columns else 0
            
            # Categorical
            try:
                debug_info = _predictor.model.debug_info()
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

            probs = _predictor.model.predict(X)
            
            df_preds = df_proc.copy()
            df_preds['probability'] = probs
            df_preds['horse_number'] = df_preds['馬番']
            
            if '単勝' in df_preds.columns:
                df_preds['odds'] = pd.to_numeric(df_preds['単勝'], errors='coerce').fillna(10.0)
            else:
                df_preds['odds'] = 10.0
            
            df_preds['expected_value'] = df_preds['probability'] * df_preds['odds']
            
        except Exception as e:
            # print(f"Proc Error {race_id}: {e}")
            return None 
            
        # 2. Allocation & Sim
        local_results = {}
        
        for strat in STRATEGIES:
            for bud in BUDGETS:
                key = (strat, bud)
                res = {'cost': 0, 'return': 0, 'hits': 0}
                
                try:
                    recs = BettingAllocator.allocate_budget(df_preds, bud, strategy=strat)
                except:
                    recs = []
                
                if recs:
                    for rec in recs:
                        cost = rec.get('total_amount', rec.get('amount', 0))
                        res['cost'] += cost
                        
                        pay = verify_hit_logic(race_id, rec, _returns_df)
                        res['return'] += pay
                        if pay > 0:
                            res['hits'] += 1
                            
                local_results[key] = res
                
        return local_results
        
    except Exception as e:
        return None

def run_parallel_simulation():
    workers = 2  # Reduce parallelism to save memory (was min(8, cpu_count))
    print(f"=== 高速シミュレーション (並列数: {workers}) ===", flush=True)
    
    # 1. Main Load & Filter
    predictor, hr, peds, df_target, returns = load_resources_main()
    
    race_ids = df_target['race_id'].unique()
    print(f"Target Races: {len(race_ids)} (2024-2025 Full)", flush=True)

    # 2. Dump temp files for workers (Correct way for Windows)
    temp_dir = os.path.join(RAW_DATA_DIR, 'temp_worker_data')
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    paths = {
        'pred': os.path.join(temp_dir, 'predictor.pkl'),
        'hr': os.path.join(temp_dir, 'hr.pkl'),
        'peds': os.path.join(temp_dir, 'peds.pkl'),
        'df': os.path.join(temp_dir, 'df_target.pkl'),
        'ret': os.path.join(temp_dir, 'returns.pkl')
    }
    
    print("Saving temp resources for workers...", flush=True)
    with open(paths['pred'], 'wb') as f: pickle.dump(predictor, f)
    with open(paths['hr'], 'wb') as f: pickle.dump(hr, f)
    with open(paths['peds'], 'wb') as f: pickle.dump(peds, f)
    with open(paths['df'], 'wb') as f: pickle.dump(df_target, f)
    with open(paths['ret'], 'wb') as f: pickle.dump(returns, f)
    
    # Clean up predictor/hr from main process memory
    del predictor, hr, peds, df_target, returns
    import gc
    gc.collect()

    total_results = {
        (strat, bud): {'cost': 0, 'return': 0, 'hits': 0, 'total': 0}
        for strat in STRATEGIES for bud in BUDGETS
    }
    
    print(f"Starting ProcessPoolExecutor...", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(paths['pred'], paths['hr'], paths['peds'], paths['df'], paths['ret'])) as executor:
            # Use chunks
            results_gen = executor.map(process_race_job, race_ids, chunksize=20)
            
            for res in tqdm(results_gen, total=len(race_ids), mininterval=1.0):
                if res:
                    for key, val in res.items():
                        total_results[key]['cost'] += val['cost']
                        total_results[key]['return'] += val['return']
                        if val['cost'] > 0: total_results[key]['total'] += 1
                        if val['return'] > 0: total_results[key]['hits'] += 1
    finally:
        # Cleanup
        try: shutil.rmtree(temp_dir)
        except: pass

    # 3. Report
    report_path = 'simulation_report_parallel.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 戦略比較シミュレーションレポート (Parallel)\n\n")
        f.write(f"- 期間: 2024-2025 ({len(race_ids)}レース)\n\n")
        f.write("| 戦略 | 予算 | 投資額 | 回収額 | 回収率 | 的中レース数 | 投資レース数 |\n")
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
    multiprocessing.freeze_support()
    run_parallel_simulation()
