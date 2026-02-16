"""
新戦略 `meta_optimized` 検証シミュレーション
- formation_flex (既存) vs meta_optimized (新規) の比較
- 10%サンプルで実行
"""
import os
import sys
import json
import pickle
import random
import argparse
import gc
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
TARGET_YEAR = 2025
BUDGET = 5000
MIN_CONFIDENCE = 'A'

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

class SimulationLogger:
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Simulation Log Started: {self.start_time.isoformat()} ===\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: {message}\n")

def load_resources():
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f:
        returns = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    df_target = results[results['race_id'].astype(str).str.startswith('2025')].copy()
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    returns_dict = {k: v for k, v in returns.groupby('race_id_str')}
    
    del results
    gc.collect()
    
    return df_target, hr, peds, returns_dict

def load_predictor():
    model = HorseRaceModel()
    model.load(os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl'))
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    return RacePredictor(model, processor, engineer)

def process_race(race_df, predictor, hr, peds, logger):
    try:
        df = race_df.copy()
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

        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        df_res['odds'] = pd.to_numeric(df_res.get('単勝', 10), errors='coerce').fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
    except Exception as e:
        logger.error(f"process_race failed: {str(e)}")
        return None

def calculate_confidence(race_df):
    if race_df.empty: return 'D'
    top = race_df.sort_values('probability', ascending=False).iloc[0]
    top_prob = top['probability']
    top_ev = top['expected_value']
    
    if top_prob >= 0.5 or top_ev >= 1.5: return 'S'
    elif top_prob >= 0.4 or top_ev >= 1.2: return 'A'
    elif top_prob >= 0.3 or top_ev >= 1.0: return 'B'
    elif top_prob >= 0.2: return 'C'
    else: return 'D'

def verify_hit(race_id, rec, returns_dict, logger):
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None: return 0
    payout = 0
    bet_type = rec.get('type') or rec.get('bet_type')
    bet_horse_nums = set(rec.get('horses', []) or rec.get('horse_numbers', []))
    if not bet_type or not bet_horse_nums: return 0
    
    try:
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
                    if method in ['流し', 'NAGASHI', '1軸流し', 'FORMATION'] and formation and len(formation) >= 2:
                        axis = set(formation[0])
                        opponents = set(formation[1])
                        win_set = set(win_nums)
                        if axis.issubset(win_set):
                            remaining = win_set - axis
                            if remaining.issubset(opponents): is_hit = True
                    elif method == 'BOX':
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                    else:
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                
                if is_hit:
                    unit = rec.get('unit_amount', 100)
                    if unit == 0:
                        cnt = rec.get('count', rec.get('points', 0))
                        if cnt > 0: unit = rec.get('amount', rec.get('total_amount', 0)) // cnt
                        else: unit = 100
                    payout += money * (unit / 100)
            except: pass
    except Exception as e:
        logger.error(f"verify_hit failed: {str(e)}")
    return payout

def run_simulation(strategy_name: str, predictor, hr, peds, df_target, returns_dict, race_ids, logger):
    total_bet = 0
    total_return = 0
    race_count = 0
    bet_count = 0
    hit_count = 0
    CONFIDENCE_ORDER = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    logger.log(f"--- Strategy: {strategy_name} ---")
    
    for i, race_id in enumerate(race_ids):
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty: continue
            
            df_pred = process_race(race_df, predictor, hr, peds, logger)
            if df_pred is None or df_pred.empty: continue
            
            conf = calculate_confidence(df_pred)
            if CONFIDENCE_ORDER.get(conf, 99) > CONFIDENCE_ORDER.get(MIN_CONFIDENCE, 1): continue
            
            recs = BettingAllocator.allocate_budget(df_pred, BUDGET, strategy=strategy_name)
            if not recs: continue
            
            race_count += 1
            race_hit = 0
            
            for rec in recs:
                bet_amount = rec.get('total_amount', rec.get('amount', 100))
                total_bet += bet_amount
                bet_count += 1
                ret = verify_hit(race_id, rec, returns_dict, logger)
                total_return += ret
                if ret > 0: 
                    hit_count += 1
                    race_hit += ret
            
            if race_hit > 0:
                logger.log(f"HIT: {race_id} - ¥{int(race_hit):,}")
                
        except Exception as e:
            logger.error(f"Race {race_id} error: {str(e)}")
            continue
            
    recovery = (total_return / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0
    
    return {
        'strategy': strategy_name,
        'race_count': race_count,
        'bet_count': bet_count,
        'hit_count': hit_count,
        'hit_rate': hit_rate,
        'recovery_rate': recovery,
        'total_bet': total_bet,
        'total_return': total_return,
        'profit': total_return - total_bet
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimulationLogger(f"sim_new_strategy_{timestamp}.log")
    
    logger.log("Loading data...")
    df_target, hr, peds, returns_dict = load_resources()
    
    # 10%サンプル
    race_ids = df_target['race_id'].unique().tolist()
    sample_size = max(1, int(len(race_ids) * 0.1))
    race_ids = random.sample(race_ids, sample_size)
    logger.log(f"Sample: {len(race_ids)} races")
    
    predictor = load_predictor()
    
    # 比較実行
    res_old = run_simulation("formation_flex", predictor, hr, peds, df_target, returns_dict, race_ids, logger)
    logger.log(f"formation_flex: {res_old['recovery_rate']:.1f}% (+¥{res_old['profit']:,})")
    
    res_new = run_simulation("meta_optimized", predictor, hr, peds, df_target, returns_dict, race_ids, logger)
    logger.log(f"meta_optimized: {res_new['recovery_rate']:.1f}% (+¥{res_new['profit']:,})")
    
    # 結果表示
    print("\n" + "="*50)
    print("新戦略 検証結果 (10%サンプル)")
    print("="*50)
    print(f"{'指標':<15} {'formation_flex':>15} {'meta_optimized':>15}")
    print("-" * 50)
    print(f"{'回収率':<15} {res_old['recovery_rate']:>14.1f}% {res_new['recovery_rate']:>14.1f}%")
    print(f"{'収支':<15} ¥{res_old['profit']:>14,} ¥{res_new['profit']:>14,}")
    print(f"{'的中率':<15} {res_old['hit_rate']:>14.1f}% {res_new['hit_rate']:>14.1f}%")
    print(f"{'レース数':<15} {res_old['race_count']:>15} {res_new['race_count']:>15}")
    print(f"{'ベット数':<15} {res_old['bet_count']:>15} {res_new['bet_count']:>15}")
    print("-" * 50)
    
    diff = res_new['recovery_rate'] - res_old['recovery_rate']
    if diff > 0:
        print(f"\n✓ 新戦略が {diff:.1f}% 優れています")
    else:
        print(f"\n✗ 既存戦略が {abs(diff):.1f}% 優れています")

if __name__ == "__main__":
    main()
