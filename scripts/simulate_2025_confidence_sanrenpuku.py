"""
2025年 自信度別シミュレーション (3連複1軸流し / 予算1000円)
- モデル: models/historical_2010_2024 (リーク対策済み)
- 自信度算出: app.py のロジックに準拠
- 修正: returnsデータを辞書化して高速アクセス保証
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm
from collections import defaultdict

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# --- 設定 ---
STRATEGY = 'sanrenpuku_1axis'
BUDGET = 1000
MODEL_BASE_DIR = os.path.join('models', 'historical_2010_2024')
MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'model.pkl')
BATCH_SIZE = 50

def load_resources():
    print("Loading resources...", flush=True)
    import gc
    
    # 1. レース結果データのロードと2025年の抽出
    path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    with open(path, 'rb') as f: results = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    # 2025年抽出
    df_target = results[results['race_id'].astype(str).str.startswith('2025')].copy()
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    del results
    gc.collect()
    print(f"Target Races (2025): {df_target['race_id'].nunique()}", flush=True)

    # 2. 馬成績データ
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    gc.collect() 
    
    # 3. その他
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)
    
    # Pre-process returns to dict for safe access
    print("Pre-processing returns to dict...", flush=True)
    # インデックスからrace_idを抽出 (Tuple Index or MultiIndex)
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    
    # 辞書化 {race_id: dataframe}
    returns_dict = {k: v for k, v in returns.groupby('race_id_str')}
    
    del returns
    gc.collect()
    print(f"Returns dict created. Keys: {len(returns_dict)}")

    # 4. モデル
    model = HorseRaceModel()
    model.load(MODEL_PATH)
    
    try:
        with open(os.path.join(MODEL_BASE_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
        with open(os.path.join(MODEL_BASE_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    except:
        from modules.preprocessing import DataProcessor, FeatureEngineer
        processor = DataProcessor()
        engineer = FeatureEngineer()

    predictor = RacePredictor(model, processor, engineer)
    return predictor, hr, peds, df_target, returns_dict

def process_batch_safe(df_batch, predictor, hr, peds):
    """データリークを防止したバッチ予測"""
    try:
        df = df_batch.copy()
        df.columns = df.columns.str.replace(' ', '')
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        df_proc = predictor.processor.process_results(df)
        df_proc = predictor.engineer.add_horse_history_features(df_proc, hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)

        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            if c in df_proc.columns:
                X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
            else:
                X[c] = 0
            
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

        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
    except Exception:
        traceback.print_exc()
        return None

def calculate_confidence(race_df):
    """自信度を算出 (app.py準拠)"""
    if race_df.empty: return 'D'
    
    # 予測スコア順にソート済みであることを前提とするか、ここでソートする
    df_sorted = race_df.sort_values('probability', ascending=False)
    top = df_sorted.iloc[0]
    
    top_prob = top['probability']
    top_ev = top['expected_value']
    
    if top_prob >= 0.5 or top_ev >= 1.5: return 'S'
    elif top_prob >= 0.4 or top_ev >= 1.2: return 'A'
    elif top_prob >= 0.3 or top_ev >= 1.0: return 'B'
    elif top_prob >= 0.2: return 'C'
    else: return 'D'

def verify_hit_logic(race_id, rec, returns_dict):
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None:
        return 0
    
    payout = 0
    # FIX: Use .get() to avoid KeyError if 'type' is missing
    bet_type = rec.get('type') 
    if not bet_type: bet_type = rec.get('bet_type')

    bet_horse_nums = set(rec.get('horses', []))
    if not bet_horse_nums: bet_horse_nums = set(rec.get('horse_numbers', []))
    
    hits = race_rets[race_rets[0] == bet_type]

    for _, h in hits.iterrows():
        try:
            money = int(str(h[2]).replace(',','').replace('円',''))
            win_str = str(h[1]).replace('→','-')
            if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
            else: win_nums = [int(win_str)]
            
            is_hit = False
            if bet_type in ['単勝', '複勝']:
                 if win_nums[0] in bet_horse_nums: is_hit = True
            elif bet_type in ['馬連', 'ワイド', '3連複', '3連単']:
                 method = rec.get('method')
                 formation = rec.get('formation')
                 
                 if method in ['流し', 'NAGASHI', '1軸流し'] and formation and len(formation) >= 2:
                     # 軸と相手
                     axis = set(formation[0])
                     opponents = set(formation[1])
                     
                     win_set = set(win_nums)
                     # 軸が含まれているか
                     if axis.issubset(win_set):
                         # 残りの当選馬が相手に含まれているか
                         remaining = win_set - axis
                         if remaining.issubset(opponents):
                            is_hit = True
                 elif method == 'BOX':
                     if set(win_nums).issubset(bet_horse_nums): is_hit = True
                 else:
                     if set(win_nums).issubset(bet_horse_nums): is_hit = True
                
            if is_hit:
                unit = rec.get('unit_amount', 100)
                if unit == 0:
                    cnt = rec.get('count', 0)
                    if cnt > 0: unit = rec.get('amount', 0) // cnt
                    else: unit = 100 # fallback
                
                pay = money * (unit / 100)
                payout += pay

        except Exception:
            pass
    return payout

def run_simulation():
    print(f"=== 2025年 自信度別シミュレーション ({STRATEGY} / @{BUDGET}円) ===", flush=True)
    
    predictor, hr, peds, df_target, returns_dict = load_resources()
    
    race_ids = sorted(df_target['race_id'].unique())
    race_chunks = [race_ids[i:i + BATCH_SIZE] for i in range(0, len(race_ids), BATCH_SIZE)]
    
    # Aggregation: Confidence -> {cost, return, hits, races}
    agg = defaultdict(lambda: {'cost': 0, 'return': 0, 'hits': 0, 'races': 0})
    
    print(f"Processing {len(race_chunks)} batches...", flush=True)
    
    cnt = 0
    for chunk in tqdm(race_chunks):
        cnt += 1
        # if cnt > 1: break # DEBUG: Stop early
        
        df_chunk = df_target[df_target['race_id'].isin(chunk)].copy()
        df_preds = process_batch_safe(df_chunk, predictor, hr, peds)
        
        if df_preds is None:
            continue
        
        grouped = df_preds.groupby('race_id')
        for race_id, race_df in grouped:
            if len(race_df) < 5: continue
            
            # 自信度判定
            conf = calculate_confidence(race_df)
            
            try:
                # 予算配分
                recs = BettingAllocator.allocate_budget(race_df, BUDGET, strategy=STRATEGY)
                
                if recs:
                    agg[conf]['races'] += 1
                    
                    race_cost = 0
                    race_return = 0
                    
                    for rec in recs:
                        cost = rec.get('amount', rec.get('total_amount', 0))
                        
                        if 'unit_amount' not in rec:
                            c = rec.get('count', rec.get('points', 1))
                            if c > 0: rec['unit_amount'] = cost // c
                            else: rec['unit_amount'] = 0
                            
                        agg[conf]['cost'] += cost
                        race_cost += cost
                        
                        pay = verify_hit_logic(race_id, rec, returns_dict)
                        agg[conf]['return'] += pay
                        race_return += pay
                        if pay > 0: agg[conf]['hits'] += 1
                    
                    # Store details for A and B
                    if conf == 'A':
                        agg[conf].setdefault('details', []).append({
                            'race_id': race_id,
                            'cost': race_cost,
                            'return': race_return,
                            'profit': race_return - race_cost
                        })

            except Exception:
                traceback.print_exc()
                pass

    # レポート出力
    print("\n=== Confidence Level Analysis (2025 / Sanrenpuku 1-Axis / 1000yen) ===")
    print(f"{'CONF':<5} {'RACES':<6} {'COST':<10} {'RETURN':<10} {'REC_RATE':<8} {'HIT_RATE':<8}")
    print("-" * 60)
    
    total_cost = 0
    total_return = 0
    total_races = 0
    total_hits = 0
    
    # S, A, B, C, D order
    for conf in ['S', 'A', 'B', 'C', 'D']:
        d = agg[conf]
        cost = d['cost']
        ret = d['return']
        races = d['races']
        hits = d['hits']
        
        rec_rate = (ret / cost * 100) if cost > 0 else 0
        hit_rate = (hits / races * 100) if races > 0 else 0 
        
        print(f"{conf:<5} {races:<6} {cost:>10,.0f} {ret:>10,.0f} {rec_rate:>7.1f}% {hit_rate:>7.1f}%")
        
        total_cost += cost
        total_return += ret
        total_races += races
        total_hits += hits
        
    print("-" * 60)
    

    # Write DETAILS for A to file
    if 'details' in agg['A']:
        a_races = agg['A']['details']
        a_races.sort(key=lambda x: x['return'], reverse=True)
        with open('confidence_A_details.txt', 'w', encoding='utf-8') as f:
            f.write("=== Confidence A Details ===\n")
            for r in a_races:
                f.write(f"Race {r['race_id']}: Cost {r['cost']} -> Return {r['return']} (Profit {r['profit']})\n")
            
    print("-" * 60)
    total_rec_rate = (total_return / total_cost * 100) if total_cost > 0 else 0
    total_hit_rate = (total_hits / total_races * 100) if total_races > 0 else 0
    print(f"{'TOTAL':<5} {total_races:<6} {total_cost:>10,.0f} {total_return:>10,.0f} {total_rec_rate:>7.1f}% {total_hit_rate:>7.1f}%")
    
    # Save to file
    with open('simulation_report_2025_confidence_1000yen.md', 'w', encoding='utf-8') as f:
        f.write("# 2025年 自信度別シミュレーション結果\n\n")
        f.write(f"- 戦略: 3連複1軸流し (1000円)\n")
        f.write(f"- モデル: 2010-2024 Historical\n\n")
        f.write("| 自信度 | レース数 | 投資額 | 回収額 | 回収率 | 的中率 |\n")
        f.write("|:---:|---:|---:|---:|---:|---:|\n")
        for conf in ['S', 'A', 'B', 'C', 'D']:
             d = agg[conf]
             cost = d['cost']
             ret = d['return']
             races = d['races']
             hits = d['hits']
             rec_rate = (ret / cost * 100) if cost > 0 else 0
             hit_rate = (hits / races * 100) if races > 0 else 0
             f.write(f"| **{conf}** | {races:,} | ¥{cost:,.0f} | ¥{ret:,.0f} | **{rec_rate:.1f}%** | {hit_rate:.1f}% |\n")
        f.write(f"| **TOTAL** | {total_races:,} | ¥{total_cost:,.0f} | ¥{total_return:,.0f} | **{total_rec_rate:.1f}%** | {total_hit_rate:.1f}% |\n")

if __name__ == "__main__":
    run_simulation()
