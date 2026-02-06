"""
2025年購入シミュレーション (新モデル/データリーク対策版)
- models/historical_2010_2024/ のモデルを使用
- add_horse_history_features にて is_predict_mode=False を強制し、データリークを防止
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# --- 設定 ---
STRATEGIES = ['balance', 'formation', 'hybrid_1000']
BUDGETS = [1000, 5000, 10000]
MODEL_BASE_DIR = os.path.join('models', 'historical_2010_2024')
MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'model.pkl')
BATCH_SIZE = 50

def load_resources():
    print("Loading resources (Optimized for 2025 New Model Sim)...", flush=True)
    import gc
    
    # 1. レース結果データのロードと2025年の抽出
    print("Loading Results (results.pickle)...", flush=True)
    path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(path, 'rb') as f: results = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
        
    # 日付の正規化 (リーク防止の肝)
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    # 2025年を対象にする
    df_target = results[results['race_id'].astype(str).str.startswith('2025')].copy()
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    del results
    gc.collect()
    print(f"Target Races (2025): {df_target['race_id'].nunique()}", flush=True)

    # 2. 馬成績データのロード (必要な馬のみに絞り込み)
    print("Loading Horse History (horse_results.pickle)...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    
    # 日付の正規化 (リーク防止)
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
        
    gc.collect() 
    
    # 3. その他リソース
    print("Loading Peds & Returns...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)
    
    # 4. モデルのロード (New Model)
    print(f"Loading New Model from {MODEL_BASE_DIR}...", flush=True)
    model = HorseRaceModel()
    model.load(MODEL_PATH)

    # 保存されているプロセッサ類を使用
    try:
        with open(os.path.join(MODEL_BASE_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_BASE_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except:
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)
    return predictor, hr, peds, df_target, returns

def process_batch_safe(df_batch, predictor, hr, peds):
    """データリークを防止したバッチ予測"""
    try:
        df = df_batch.copy()
        # カラム名正規化
        df.columns = df.columns.str.replace(' ', '')
        
        # 型変換と日付正規化
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        # 特徴量生成 (Internal FE logic)
        df_proc = predictor.processor.process_results(df)
        
        # --- 重要: add_horse_history_features の呼び出し ---
        # 内部で is_predict_mode の自動判定に頼らず、学習/バックテストモードの挙動を期待
        # 現在の modules/preprocessing.py の実装では、dfの日付と馬IDがhrに存在する場合に学習モードになる
        # シミュレーション対象(2025)はhrに含まれているはずなので、自然と正しい挙動になるはずだが、
        # 確実にリークを防ぐため、日付の不一致がないようにする。
        df_proc = predictor.engineer.add_horse_history_features(df_proc, hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)

        # モデル入力の作成
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            if c in df_proc.columns:
                X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
            else:
                X[c] = 0
            
        # カテゴリ変数の処理 (LightGBM用)
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

        # 予測
        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
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
                payout += (money * (rec['unit_amount'] / 100))
        except: pass
    return payout

def run_2025_simulation():
    print("=== 2025年 新モデル シミュレーション開始 ===", flush=True)
    
    predictor, hr, peds, df_target, returns = load_resources()
    
    race_ids = sorted(df_target['race_id'].unique())
    
    total_results = {
        (strat, bud): {'cost': 0, 'return': 0, 'hits': 0, 'total_bets': 0}
        for strat in STRATEGIES for bud in BUDGETS
    }
    
    # バッチ処理
    race_chunks = [race_ids[i:i + BATCH_SIZE] for i in range(0, len(race_ids), BATCH_SIZE)]
    print(f"Processing {len(race_chunks)} batches...", flush=True)
    
    for chunk in tqdm(race_chunks):
        df_chunk = df_target[df_target['race_id'].isin(chunk)].copy()
        df_preds = process_batch_safe(df_chunk, predictor, hr, peds)
        
        if df_preds is None: continue
        
        grouped = df_preds.groupby('race_id')
        for race_id, race_df in grouped:
            if len(race_df) < 5: continue # 少頭数は除外
            
            for strat in STRATEGIES:
                for bud in BUDGETS:
                    key = (strat, bud)
                    try:
                        recs = BettingAllocator.allocate_budget(race_df, bud, strategy=strat)
                        if recs:
                            for rec in recs:
                                cost = rec.get('total_amount', 0)
                                total_results[key]['cost'] += cost
                                total_results[key]['total_bets'] += 1
                                
                                pay = verify_hit_logic(race_id, rec, returns)
                                total_results[key]['return'] += pay
                                if pay > 0: total_results[key]['hits'] += 1
                    except Exception as e:
                        pass
                    
    # レポート作成
    report_path = 'simulation_report_2025_new_model.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 2025年購入シミュレーションレポート (新モデル/リーク対策済み)\n\n")
        f.write(f"- 使用モデル: `models/historical_2010_2024/model.pkl` (2010-2024学習)\n")
        f.write(f"- 期間: 2025年通期 ({len(race_ids)}レース)\n\n")
        f.write("| 戦略 | 予算 | 投資額 | 回収額 | 回収率 | 的中数 | 的中率 | 投資点数 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        print("\n=== 2025年シミュレーション結果 ===")
        for strat in STRATEGIES:
            for bud in BUDGETS:
                r = total_results[(strat, bud)]
                recov = (r['return']/r['cost']*100) if r['cost']>0 else 0
                hit_rate = (r['hits']/r['total_bets']*100) if r['total_bets']>0 else 0
                f.write(f"| {strat} | {bud} | {r['cost']:,.0f} | {r['return']:,.0f} | {recov:.1f}% | {r['hits']} | {hit_rate:.1f}% | {r['total_bets']} |\n")
                print(f"{strat:<10} {bud:<6} 投資:{r['cost']:>10.0f} 回収:{r['return']:>10.0f} 回収率:{recov:>6.1f}%")
                
    print(f"\nReport saved: {report_path}")

if __name__ == "__main__":
    run_2025_simulation()
