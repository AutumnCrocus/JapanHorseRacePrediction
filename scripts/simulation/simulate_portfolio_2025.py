
"""
2025年通年 ポートフォリオ戦略検証シミュレーション
- formation_flex (攻め) + meta_contrarian (守り/一撃) の併用効果を検証
- 全レース対象
"""
import os
import sys
import pickle
import argparse
import gc
from datetime import datetime
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
TARGET_YEAR = 2025
BUDGET_PER_STRATEGY = 5000 # 各戦略5000円 (計1万円/レース)
MIN_CONFIDENCE = 'A' # Aランク以上のみ参加など必要なら設定 (今回は全レース対象とするか、Allocatorに任せる)
# Allocatorはスコアに応じて「参加しない」判断も持っているため、ここではフィルタしない

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

class SimulationLogger:
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Portfolio Simulation Started: {self.start_time.isoformat()} ===\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            
    def error(self, message: str):
        self.log(f"ERROR: {message}")

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
    
    # Filter 2025
    df_target = results[results['race_id'].astype(str).str.startswith(str(TARGET_YEAR))].copy()
    
    # Filter HR data for memory efficiency (optional but good practice)
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    
    # Prepare returns dict for fast lookup
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    returns_dict = {k: v for k, v in returns.groupby('race_id_str')}
    
    del results
    gc.collect()
    
    return df_target, hr, peds, returns_dict

def load_predictor():
    print("Loading AI model...")
    model = HorseRaceModel()
    
    # Use historical model to avoid leakage if testing 2025
    # Assuming historical_2010_2024 was trained on data UP TO 2024
    model_path = os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        print(f"Warning: Historical model not found at {model_path}. Using current model.")
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
    rec_type = rec.get('bet_type') or rec.get('type') # Handle both keys
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
                
                # Logic copied from verify_hit in simulate_meta_contrarian.py
                # Simplified for checking correctness
                
                if rec_type in ['単勝', '複勝']:
                    if win_nums[0] in bet_horse_nums: is_hit = True
                elif rec_type in ['馬連', 'ワイド', '3連複', '3連単']:
                    method = rec.get('method')
                    formation = rec.get('formation')
                    
                    if method in ['流し', 'Formation', 'FORMATION'] and formation:
                        if rec_type == '3連複':
                            # [[軸], [相手]] or [[1], [2], [3]] ?
                            # BettingAllocator implementation:
                            # 1-axis: [[axis], [opponents]]
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
                    elif method == 'BOX' or not method:
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                        
                if is_hit:
                    # Calculate unit amount
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
    logger = SimulationLogger(f"sim_portfolio_2025_{timestamp}.log")

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio (0.0-1.0)')
    args = parser.parse_args()

    df_target, hr, peds, returns_dict = load_resources()
    race_ids = sorted(df_target['race_id'].unique().tolist())
    
    if args.sample < 1.0:
        import random
        sample_size = int(len(race_ids) * args.sample)
        race_ids = random.sample(race_ids, sample_size)
        logger.log(f"Sampling {args.sample*100}%: {len(race_ids)} races")
    
    logger.log(f"Target Year: {TARGET_YEAR}")
    logger.log(f"Total Races: {len(race_ids)}")
    logger.log(f"Strategies: formation_flex + meta_contrarian")
    logger.log(f"Budget per strategy: {BUDGET_PER_STRATEGY} JPY")
    
    predictor = load_predictor()
    
    # Stats
    stats = {
        'flex': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'cont': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0},
        'portfolio': {'bet': 0, 'return': 0, 'hit': 0, 'races': 0}
    }
    
    print(f"Starting simulation for {len(race_ids)} races (Parallel)...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Wrapper for parallel execution
    def process_wrapper(race_id):
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty: return None
            
            # Predict
            df_pred = process_race(race_df, predictor, hr, peds)
            if df_pred is None or df_pred.empty: return None
            
            # 1. Formation Flex Strategy
            recs_flex = BettingAllocator.allocate_budget(df_pred, BUDGET_PER_STRATEGY, strategy='formation_flex')
            
            # 2. Meta Contrarian Strategy
            recs_cont = BettingAllocator.allocate_budget(df_pred, BUDGET_PER_STRATEGY, strategy='meta_contrarian')
            
            # Eval
            res = {
                'race_id': race_id,
                'flex': {'bet': 0, 'return': 0, 'hit': 0},
                'cont': {'bet': 0, 'return': 0, 'hit': 0},
                'portfolio': {'bet': 0, 'return': 0, 'hit': 0}
            }
            
            # Flex
            if recs_flex:
                for r in recs_flex:
                    amt = r.get('total_amount', r.get('amount', 0))
                    res['flex']['bet'] += amt
                    ret = verify_hit(r, returns_dict, race_id)
                    res['flex']['return'] += ret
                    if ret > 0: res['flex']['hit'] = 1 # Flag as hit if any returns
            
            # Cont
            if recs_cont:
                for r in recs_cont:
                    amt = r.get('total_amount', r.get('amount', 0))
                    res['cont']['bet'] += amt
                    ret = verify_hit(r, returns_dict, race_id)
                    res['cont']['return'] += ret
                    if ret > 0: res['cont']['hit'] = 1
            
            # Portfolio
            res['portfolio']['bet'] = res['flex']['bet'] + res['cont']['bet']
            res['portfolio']['return'] = res['flex']['return'] + res['cont']['return']
            if res['portfolio']['return'] > 0: res['portfolio']['hit'] = 1
            
            return res
        except Exception as e:
            return None

    # Run Parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_wrapper, rid): rid for rid in race_ids}
        
        for future in tqdm(as_completed(futures), total=len(race_ids)):
            res = future.result()
            if not res: continue
            
            # Aggregate stats
            for key in ['flex', 'cont', 'portfolio']:
                stats[key]['bet'] += res[key]['bet']
                stats[key]['return'] += res[key]['return']
                if res[key]['bet'] > 0: stats[key]['races'] += 1
                if res[key]['return'] > 0: stats[key]['hit'] += 1 # Hit counter logic changed to per-race
            
            rid = res['race_id']
            port_ret = res['portfolio']['return']
            port_bet = res['portfolio']['bet']
            
            if port_ret >= 50000:
                 logger.log(f"HIT {rid}: Invest {port_bet} -> Return {int(port_ret):,}")


    # Summary
    logger.log("\n=== Final Results ===")
    
    def print_stat(name, s):
        recov = (s['return'] / s['bet'] * 100) if s['bet'] > 0 else 0
        profit = s['return'] - s['bet']
        msg = f"{name}: Bet {s['bet']:,} -> Return {int(s['return']):,} (Recov: {recov:.1f}%, Profit: {int(profit):,})"
        logger.log(msg)
        print(msg)
        
    print("\n")
    print_stat("Formation Flex", stats['flex'])
    print_stat("Meta Contrarian", stats['cont'])
    print_stat("Portfolio Total", stats['portfolio'])
    
    logger.log("Simulation Completed.")

if __name__ == "__main__":
    main()
