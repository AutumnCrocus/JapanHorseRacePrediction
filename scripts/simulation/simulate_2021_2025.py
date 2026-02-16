
"""
2021-2025 長期シミュレーション
- モデル: models/historical_2010_2020/ (2010-2020学習)
- 期間: 2021-01-01 ~ 2025-12-31
- 予算: 5000円/レース
- 全戦略比較
"""
import os
import sys
import pickle
import argparse
import gc
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
START_YEAR = 2021
END_YEAR = 2025
BUDGET = 5000
MODEL_PATH_BASE = os.path.join(MODEL_DIR, "historical_2010_2020")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Strategies
STRATEGIES = [
    'box4_sanrenpuku',
    'box4_umaren',
    'meta_contrarian',
    'formation_flex',
    'sanrenpuku_2axis',
    'sanrenpuku_1axis',
    'formation',
    'umaren_nagashi',
    'wide_nagashi',
    'hybrid_1000'
]

class SimulationLogger:
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== 2021-2025 Simulation Started: {self.start_time.isoformat()} ===\n")
            f.write(f"Model: {MODEL_PATH_BASE}\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            
def load_resources():
    print("Loading data resources...")
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
    
    # Filter for target years
    df_target = results[(results['date'].dt.year >= START_YEAR) & (results['date'].dt.year <= END_YEAR)].copy()
    
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    returns_dict = {k: v for k, v in returns.groupby('race_id_str')}
    
    del results
    gc.collect()
    
    return df_target, hr, peds, returns_dict

def load_predictor():
    print("Loading AI model...")
    model = HorseRaceModel()
    
    model_path = os.path.join(MODEL_PATH_BASE, 'model.pkl')
    proc_path = os.path.join(MODEL_PATH_BASE, 'processor.pkl')
    eng_path = os.path.join(MODEL_PATH_BASE, 'engineer.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train model first.")
        
    model.load(model_path)
    with open(proc_path, 'rb') as f: processor = pickle.load(f)
    with open(eng_path, 'rb') as f: engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def batch_process_features(df_target, predictor, hr, peds):
    print("Batch processing features...", flush=True)
    df = df_target.copy()
    
    # Basic preprocessing (same as before)
    df.columns = df.columns.str.replace(' ', '')
    for col in ['馬番', '枠番', '単勝']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    # Feature Engineering (Batch)
    try:
        df_proc = predictor.processor.process_results(df)
        df_proc = predictor.engineer.add_horse_history_features(df_proc, hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)
        
        # Add probability
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            if c in df_proc.columns:
                col_data = df_proc[c]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                X[c] = pd.to_numeric(col_data, errors='coerce').fillna(0)
            else:
                X[c] = 0
                
        probs = predictor.model.predict(X)
        df_proc['probability'] = probs
        
        # Prepare for betting
        df_proc['horse_number'] = df_proc['馬番']
        df_proc['odds'] = pd.to_numeric(df_proc.get('単勝', 10), errors='coerce').fillna(10.0)
        df_proc['expected_value'] = df_proc['probability'] * df_proc['odds']
        
        return df_proc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def verify_hit(rec, returns_dict, race_id):
    # Same logic as before
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None: return 0
    payout = 0
    rec_type = rec.get('bet_type') or rec.get('type')
    
    bet_horse_nums = set(rec.get('horse_numbers') or rec.get('horses', []))
    if not rec_type or not bet_horse_nums: return 0
    
    try:
        hits = race_rets[race_rets[0] == rec_type]
        for _, h in hits.iterrows():
            try:
                money = int(str(h[2]).replace(',','').replace('円',''))
                win_str = str(h[1]).replace('→','-')
                if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
                else: win_nums = [int(win_str)]
                is_hit = False
                
                method = rec.get('method')
                formation = rec.get('formation')
                
                if rec_type in ['単勝', '複勝']:
                    if win_nums[0] in bet_horse_nums: is_hit = True
                elif rec_type in ['馬連', 'ワイド', '3連複', '3連単']:
                    if method in ['流し', 'Formation', 'FORMATION'] and formation:
                        if rec_type == '3連複':
                            if len(formation) == 2:
                                axis = set(formation[0])
                                opponents = set(formation[1])
                                win_set = set(win_nums)
                                if axis.issubset(win_set):
                                    if (win_set - axis).issubset(opponents): is_hit = True
                        elif rec_type == '3連単':
                            if len(formation) == 3:
                                g1, g2, g3 = [set(x) for x in formation]
                                if win_nums[0] in g1 and win_nums[1] in g2 and win_nums[2] in g3:
                                    is_hit = True
                        elif rec_type == '馬連' or rec_type == 'ワイド':
                            if len(formation) == 2:
                                head = set(formation[0])
                                opps = set(formation[1])
                                if not head.isdisjoint(set(win_nums)):
                                     other = set(win_nums) - head
                                     if other.issubset(opps): 
                                         is_hit = True

                    elif method == 'BOX' or not method:
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                        
                if is_hit:
                    unit = rec.get('unit_amount', 100)
                    if unit == 0:
                        count = rec.get('points', rec.get('count', 1))
                        total_amt = rec.get('total_amount', rec.get('amount', 0))
                        unit = total_amt // count if count > 0 else 100
                    payout += money * (unit / 100)
            except: pass
    except: pass
    return payout

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimulationLogger(f"sim_2021_2025_{timestamp}.log")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio (0.0-1.0)')
    args = parser.parse_args()

    df_target, hr, peds, returns_dict = load_resources()
    
    # Sample race_ids FIRST
    all_race_ids = sorted(df_target['race_id'].unique().tolist())
    if args.sample < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(all_race_ids) * args.sample)
        race_ids = random.sample(all_race_ids, sample_size)
        logger.log(f"Sampling {args.sample*100}%: {len(race_ids)} races")
    else:
        race_ids = all_race_ids
        
    logger.log(f"Total Target Races: {len(race_ids)}")
    
    # Filter df_target to sampled races (Efficiency)
    df_sampled = df_target[df_target['race_id'].isin(race_ids)].copy()
    
    predictor = load_predictor()
    
    # BATCH PROCESS
    try:
        df_processed = batch_process_features(df_sampled, predictor, hr, peds)
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return

    # Yearly stats storage
    yearly_stats = {y: {s: {'bet': 0, 'return': 0, 'hit': 0, 'races': 0} for s in STRATEGIES} for y in range(START_YEAR, END_YEAR+1)}
    
    # Iterate by race_id
    print(f"Simulating betting for {len(race_ids)} races...", flush=True)
    
    # Groupby is efficient
    grouped = df_processed.groupby('race_id')
    
    for race_id, df_pred in tqdm(grouped, total=len(grouped)):
        try:
            race_year = int(str(race_id)[:4])
            
            for strat in STRATEGIES:
                budget = 1000 if strat == 'hybrid_1000' else BUDGET
                recs = BettingAllocator.allocate_budget(df_pred, budget, strategy=strat)
                
                s_bet = 0
                s_ret = 0
                s_hit = 0
                
                if recs:
                    for r in recs:
                        amt = r.get('total_amount', r.get('amount', 0))
                        s_bet += amt
                        ret = verify_hit(r, returns_dict, race_id)
                        s_ret += ret
                        if ret > 0: s_hit = 1
                
                yearly_stats[race_year][strat]['bet'] += s_bet
                yearly_stats[race_year][strat]['return'] += s_ret
                if s_bet > 0: yearly_stats[race_year][strat]['races'] += 1
                if s_hit > 0: yearly_stats[race_year][strat]['hit'] += 1
        except Exception as e:
            print(f"Error in race {race_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Report & Log
    final_stats = {s: {'bet': 0, 'return': 0} for s in STRATEGIES}
    
    for year in range(START_YEAR, END_YEAR+1):
        logger.log(f"\n=== Year {year} Results ===")
        print(f"\n--- {year} ---")
        
        y_res = []
        for s in STRATEGIES:
            d = yearly_stats[year][s]
            final_stats[s]['bet'] += d['bet']
            final_stats[s]['return'] += d['return']
            
            recov = (d['return'] / d['bet'] * 100) if d['bet'] > 0 else 0
            profit = d['return'] - d['bet']
            y_res.append({'Strategy': s, 'Recovery': recov, 'Profit': profit, 'Bet': d['bet']})
            
        df_y = pd.DataFrame(y_res).sort_values('Recovery', ascending=False)
        print(df_y.to_string(index=False))
        for _, row in df_y.iterrows():
             logger.log(f"{row['Strategy']:<20} | {row['Recovery']:6.1f}% | {row['Profit']:10,}")

    logger.log(f"\n=== Total Results ({START_YEAR}-{END_YEAR}) ===")
    print(f"\n=== Total ({START_YEAR}-{END_YEAR}) ===")
    
    t_res = []
    for s in STRATEGIES:
        d = final_stats[s]
        recov = (d['return'] / d['bet'] * 100) if d['bet'] > 0 else 0
        profit = d['return'] - d['bet']
        t_res.append({'Strategy': s, 'Recovery': recov, 'Profit': profit, 'Bet': d['bet']})
        
    df_t = pd.DataFrame(t_res).sort_values('Recovery', ascending=False)
    print(df_t.to_string(index=False))
    
    for _, row in df_t.iterrows():
         logger.log(f"{row['Strategy']:<20} | {row['Recovery']:6.1f}% | {row['Profit']:10,}")
         
    logger.log("Simulation Completed.")

if __name__ == "__main__":
    main()
