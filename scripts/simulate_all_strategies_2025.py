
"""
2025年通年 全戦略比較シミュレーション
- 既存の全戦略を同一条件(10%サンプル)で実行し、有効なロジックを探索する
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
TARGET_YEAR = 2025
BUDGET = 5000 # 各戦略5000円
HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Test Strategies
STRATEGIES = [
    'formation', 
    'formation_flex', 
    'meta_contrarian',
    'box4_sanrenpuku', 
    'sanrenpuku_1axis',
    'sanrenpuku_2axis',
    'wide_nagashi',
    'umaren_nagashi',
    'box4_umaren',
    'hybrid_1000' # Budget 1000 but helpful to check logic
]

class SimulationLogger:
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== All Strategies Simulation Started: {self.start_time.isoformat()} ===\n")
    
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
    
    df_target = results[results['race_id'].astype(str).str.startswith(str(TARGET_YEAR))].copy()
    
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
    
    model_path = os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        proc_path = os.path.join(MODEL_DIR, 'processor.pkl')
        eng_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    else:
        proc_path = os.path.join(HISTORICAL_MODEL_DIR, 'processor.pkl')
        eng_path = os.path.join(HISTORICAL_MODEL_DIR, 'engineer.pkl')
        
    model.load(model_path)
    with open(proc_path, 'rb') as f: processor = pickle.load(f)
    with open(eng_path, 'rb') as f: engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def process_race(race_df, predictor, hr, peds):
    df = race_df.copy()
    df.columns = df.columns.str.replace(' ', '')
    for col in ['馬番', '枠番', '単勝']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    try:
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
        return None

def verify_hit(rec, returns_dict, race_id):
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
                if rec_type in ['単勝', '複勝']:
                    if win_nums[0] in bet_horse_nums: is_hit = True
                elif rec_type in ['馬連', 'ワイド', '3連複', '3連単']:
                    method = rec.get('method')
                    formation = rec.get('formation')
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
                                # Normal Nagashi Logic for Umaren/Wide
                                # formation = [[head], [opps]]
                                head = set(formation[0])
                                opps = set(formation[1])
                                # Standard Logic check
                                # Check if head is in win_nums (size 2)
                                # And check if other in win_nums is in opps
                                if not head.isdisjoint(set(win_nums)):
                                     other = set(win_nums) - head
                                     if other.issubset(opps): is_hit = True

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
    logger = SimulationLogger(f"sim_all_strategies_2025_{timestamp}.log")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio')
    args = parser.parse_args()

    df_target, hr, peds, returns_dict = load_resources()
    race_ids = sorted(df_target['race_id'].unique().tolist())
    
    if args.sample < 1.0:
        import random
        random.seed(42) # Fixed seed for reproducibility
        sample_size = int(len(race_ids) * args.sample)
        race_ids = random.sample(race_ids, sample_size)
    
    logger.log(f"Target Year: {TARGET_YEAR}, Races: {len(race_ids)}, Budget: {BUDGET}")
    predictor = load_predictor()
    
    # Init Stats
    stats = {s: {'bet': 0, 'return': 0, 'hit': 0, 'races': 0} for s in STRATEGIES}
    
    print(f"Comparing {len(STRATEGIES)} strategies on {len(race_ids)} races...")
    
    def process_wrapper(race_id):
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty: return None
            df_pred = process_race(race_df, predictor, hr, peds)
            if df_pred is None or df_pred.empty: return None
            
            res_map = {}
            for strat in STRATEGIES:
                budget = 1000 if strat == 'hybrid_1000' else BUDGET
                # low budget strategies usually hardcoded inside allocator, 
                # but we pass 5000 generally. 
                # Except Sanrenpuku 1axis (1500), etc. Allocator handles them?
                
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
                
                res_map[strat] = {'bet': s_bet, 'return': s_ret, 'hit': s_hit}
            return res_map
        except: return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_wrapper, rid): rid for rid in race_ids}
        for future in tqdm(as_completed(futures), total=len(race_ids)):
            res = future.result()
            if not res: continue
            for strat, val in res.items():
                stats[strat]['bet'] += val['bet']
                stats[strat]['return'] += val['return']
                if val['bet'] > 0: stats[strat]['races'] += 1
                if val['hit'] > 0: stats[strat]['hit'] += 1 # Count hit races

    # Report
    logger.log("\n=== Final Ranking (Recovery Rate) ===")
    results_list = []
    for s, v in stats.items():
        recov = (v['return'] / v['bet'] * 100) if v['bet'] > 0 else 0
        profit = v['return'] - v['bet']
        results_list.append({
            'Strategy': s,
            'Recovery': recov,
            'Profit': profit,
            'Bet': v['bet'],
            'Return': v['return'],
            'HitRate': (v['hit'] / v['races'] * 100) if v['races'] > 0 else 0
        })
        
    df_res = pd.DataFrame(results_list).sort_values('Recovery', ascending=False)
    
    print("\n")
    print(df_res.to_string(index=False))
    
    for _, row in df_res.iterrows():
        logger.log(f"{row['Strategy']:<20} | Recov: {row['Recovery']:6.1f}% | Profit: {row['Profit']:10,} | HitRate: {row['HitRate']:5.1f}%")
        
    logger.log("Simulation Completed.")

if __name__ == "__main__":
    main()
