import os
import sys
import pickle
import argparse
import gc
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
TARGET_YEAR = 2025
BUDGET = 5000
RANKING_MODEL_DIR = os.path.join(MODEL_DIR, "standalone_ranking")
LGBM_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), '..', 'simulation_progress_full.json')
os.makedirs(LOGS_DIR, exist_ok=True)

class SimulationLogger:
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Full Comparison Simulation (Sequential) Started: {self.start_time.isoformat()} ===\n")
            f.write(f"Target Year: {TARGET_YEAR}, Budget: {BUDGET}\n")
            f.write(f"LTR Model: {RANKING_MODEL_DIR}\n")
            f.write(f"LGBM Model: {LGBM_MODEL_DIR}\n\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

def load_all_resources():
    """全てのリソースをロードする"""
    print("Loading data resources...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f:
        returns = pickle.load(f)
    
    # Pre-processing
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    df_target = results[results['race_id'].astype(str).str.startswith(str(TARGET_YEAR))].copy()
    
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    returns_dict = {k: v.to_dict('records') for k, v in returns.groupby('race_id_str')}
    
    return df_target, hr, peds, returns_dict

def load_model_resources(model_dir: str):
    """モデルリソースをロードする"""
    model_name = "ranking_model.pkl" if "ranking" in model_dir else "model.pkl"
    model_path = os.path.join(model_dir, model_name)
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        model = data.get('model', data)
        features = data.get('feature_names', [])
    else:
        model = data
        features = []
        
    proc_path = os.path.join(model_dir, 'processor.pkl')
    if not os.path.exists(proc_path): proc_path = os.path.join(MODEL_DIR, 'processor.pkl')
    eng_path = os.path.join(model_dir, 'engineer.pkl')
    if not os.path.exists(eng_path): eng_path = os.path.join(MODEL_DIR, 'engineer.pkl')

    with open(proc_path, 'rb') as f: processor = pickle.load(f)
    with open(eng_path, 'rb') as f: engineer = pickle.load(f)
    
    if not features:
        features = processor.get_feature_names() if hasattr(processor, 'get_feature_names') else []

    return model, features, processor, engineer

def process_race(rid, race_df, hr, peds, returns_for_race, context):
    """1レースの処理"""
    try:
        # LTRのリソースを使用して特徴量作成（共通）
        _, _, ltr_processor, ltr_engineer = context['ltr']
        
        df = race_df.copy()
        df.columns = df.columns.str.replace(' ', '')
        
        df_proc = ltr_processor.process_results(df)
        df_proc = ltr_engineer.add_horse_history_features(df_proc, hr)
        df_proc = ltr_engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = ltr_engineer.add_jockey_features(df_proc)
        df_proc = ltr_engineer.add_pedigree_features(df_proc, peds)
        df_proc = ltr_engineer.add_odds_features(df_proc)
        
        race_results = {}
        for m_key, strat in [('ltr', 'ranking_anchor'), ('lgbm', 'formation_flex')]:
            model, features, _, _ = context[m_key]
            
            X = pd.DataFrame(index=df_proc.index)
            feats = features if features else [c for c in df_proc.select_dtypes(include=[np.number]).columns 
                                             if c not in ['rank', 'probability', '馬番']]
            for c in feats:
                if c in df_proc.columns:
                    X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
                else:
                    X[c] = 0

            # 予測
            if m_key == 'lgbm' and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            
            df_pred = df_proc.copy()
            df_pred['probability'] = preds
            df_pred['horse_number'] = df_pred['馬番']
            df_pred['odds'] = pd.to_numeric(df_pred.get('単勝', 10), errors='coerce').fillna(10.0)
            
            recs = BettingAllocator.allocate_budget(df_pred, BUDGET, strategy=strat)
            
            bet_amt = 0; ret_amt = 0
            if recs:
                for rec in recs:
                    bet_amt += rec.get('total_amount', 0)
                    payout = 0
                    rec_type = rec.get('bet_type') or rec.get('type')
                    bet_horses = set(rec.get('horse_numbers') or rec.get('horses', []))
                    
                    if rec_type and bet_horses:
                        for h in returns_for_race:
                            if h[0] == rec_type:
                                try:
                                    money = int(str(h[2]).replace(',','').replace('円',''))
                                    win_str = str(h[1]).replace('→','-')
                                    win_nums = [int(x) for x in win_str.split('-')] if '-' in win_str else [int(win_str)]
                                    
                                    is_hit = False
                                    if rec_type in ['単勝', '複勝']:
                                        if win_nums[0] in bet_horses: is_hit = True
                                    elif rec_type in ['馬連', 'ワイド', '3連複', '3連単']:
                                        method = rec.get('method')
                                        formation = rec.get('formation')
                                        if method in ['流し', 'Formation', 'FORMATION', 'NAGASHI'] and formation:
                                            if rec_type == '3連複' and len(formation) == 2:
                                                axis = set(formation[0]); opps = set(formation[1]); win_set = set(win_nums)
                                                if axis.issubset(win_set) and (win_set - axis).issubset(opps): is_hit = True
                                            elif rec_type == '3連単' and len(formation) == 3:
                                                if win_nums[0] in formation[0] and win_nums[1] in formation[1] and win_nums[2] in formation[2]: is_hit = True
                                            elif rec_type in ['馬連', 'ワイド'] and len(formation) == 2:
                                                head = set(formation[0]); opps = set(formation[1])
                                                if not head.isdisjoint(set(win_nums)) and (set(win_nums) - head).issubset(opps): is_hit = True
                                        elif method == 'BOX' or not method:
                                            if set(win_nums).issubset(bet_horses): is_hit = True
                                    
                                    if is_hit:
                                        unit = rec.get('unit_amount', 100)
                                        payout += money * (unit / 100)
                                except: pass
                    ret_amt += payout
            
            race_results[m_key] = {'bet': bet_amt, 'return': ret_amt, 'hit': 1 if ret_amt > 0 else 0}
        return race_results
    except Exception as e:
        return {'ltr': {'bet': 0, 'return': 0, 'hit': 0}, 'lgbm': {'bet': 0, 'return': 0, 'hit': 0}}

def update_progress(current, total, stats, start_time, elapsed_prev=0):
    pct = (current / total) * 100
    elapsed_now = time.time() - start_time
    elapsed_total = elapsed_now + elapsed_prev
    etr = (elapsed_total / current) * (total - current) if current > 0 else 0
    progress = {
        'status': 'running',
        'progress_pct': round(pct, 2),
        'races_processed': current,
        'total_races': total,
        'elapsed_seconds': round(elapsed_total, 1),
        'remaining_seconds_est': round(etr, 1),
        'summary': {
            'LTR': {
                'recovery': round((stats['ltr']['return'] / stats['ltr']['bet'] * 100) if stats['ltr']['bet'] > 0 else 0, 1),
                'hit_rate': round((stats['ltr']['hit'] / stats['ltr']['races'] * 100) if stats['ltr']['races'] > 0 else 0, 1),
                'bet': stats['ltr']['bet'],
                'return': stats['ltr']['return'],
                'hit': stats['ltr']['hit'],
                'races': stats['ltr']['races']
            },
            'LGBM': {
                'recovery': round((stats['lgbm']['return'] / stats['lgbm']['bet'] * 100) if stats['lgbm']['bet'] > 0 else 0, 1),
                'hit_rate': round((stats['lgbm']['hit'] / stats['lgbm']['races'] * 100) if stats['lgbm']['races'] > 0 else 0, 1),
                'bet': stats['lgbm']['bet'],
                'return': stats['lgbm']['return'],
                'hit': stats['lgbm']['hit'],
                'races': stats['lgbm']['races']
            }
        }
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"full_comparison_seq_{timestamp}.log"
    logger = SimulationLogger(log_name)
    
    df_raw, hr, peds, returns_dict = load_all_resources()
    race_ids = sorted(df_raw['race_id'].unique().tolist())
    
    if args.sample < 1.0:
        import random
        random.seed(42)
        race_ids = sorted(random.sample(race_ids, int(len(race_ids) * args.sample)))
    
    total = len(race_ids)
    
    stats = {
        'ltr': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'lgbm': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
    }
    start_index = 0
    elapsed_prev = 0

    if args.resume and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            if progress['total_races'] == total:
                start_index = progress['races_processed']
                elapsed_prev = progress.get('elapsed_seconds', 0)
                sum_data = progress['summary']
                for k in ['ltr', 'lgbm']:
                    key_upper = k.upper()
                    stats[k]['bet'] = sum_data[key_upper].get('bet', 0)
                    stats[k]['return'] = sum_data[key_upper].get('return', 0)
                    stats[k]['hit'] = sum_data[key_upper].get('hit', 0)
                    stats[k]['races'] = sum_data[key_upper].get('races', 0)
                logger.log(f"Resuming from race index {start_index} (Previous recovery LTR: {stats['ltr'].get('recovery', 0)}%)")
            else:
                logger.log("Progress file total races mismatch, starting from scratch.")
        except Exception as e:
            logger.log(f"Failed to load progress file: {e}, starting from scratch.")

    logger.log(f"Starting sequential simulation for {total} races (index {start_index} to {total}).")
    
    context = {
        'ltr': load_model_resources(RANKING_MODEL_DIR),
        'lgbm': load_model_resources(LGBM_MODEL_DIR)
    }
    
    start_time = time.time()
    for i in tqdm(range(start_index, total)):
        rid = race_ids[i]
        race_df = df_raw[df_raw['race_id'] == rid]
        ret_for_race = returns_dict.get(str(rid), [])
        
        res = process_race(rid, race_df, hr, peds, ret_for_race, context)
        
        for key in ['ltr', 'lgbm']:
            stats[key]['bet'] += res[key]['bet']
            stats[key]['return'] += res[key]['return']
            if res[key]['bet'] > 0:
                stats[key]['races'] += 1
                stats[key]['hit'] += res[key]['hit']
        
        if (i+1) % max(1, total // 10) == 0 or (i+1) == total:
            ltr_rec = (stats['ltr']['return'] / stats['ltr']['bet'] * 100) if stats['ltr']['bet'] > 0 else 0
            lgbm_rec = (stats['lgbm']['return'] / stats['lgbm']['bet'] * 100) if stats['lgbm']['bet'] > 0 else 0
            logger.log(f"Progress: {i+1}/{total} | LTR Rec: {ltr_rec:.1f}% | LGBM Rec: {lgbm_rec:.1f}%")
        
        # 10レースごと（または必要なら毎レース）にJSONのみ更新
        if (i+1) % 10 == 0 or (i+1) == total:
            update_progress(i+1, total, stats, start_time, elapsed_prev)

    logger.log("\n" + "="*40 + "\nFINAL RESULTS (Sequential)\n" + "="*40)
    for key, name in [('ltr', 'LTR (Ranking)'), ('lgbm', 'LGBM (Historical)')]:
        v = stats[key]
        recov = (v['return'] / v['bet'] * 100) if v['bet'] > 0 else 0
        hit_rate = (v['hit'] / v['races'] * 100) if v['races'] > 0 else 0
        logger.log(f"{name:<20}: Recov: {recov:6.1f}% | Profit: {v['return']-v['bet']:10.0f} | HitRate: {hit_rate:5.1f}%")

    logger.log("Simulation Completed.")
    update_progress(total, total, stats, start_time, elapsed_prev)

if __name__ == "__main__":
    main()
