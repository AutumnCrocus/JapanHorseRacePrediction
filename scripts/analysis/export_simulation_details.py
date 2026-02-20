
"""
シミュレーション詳細データのエクスポート
- 対象: 2025年全レース
- 戦略: formation (500円), balance (1000円)
- 出力: CSV (race_id, date, venue, strategy, budget, cost, return, hit, profit, bets_count, bet_string)
"""
import os
import sys
import pandas as pd
import pickle
from tqdm import tqdm

# Set threading limits
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# Simulation settings
TARGET_STRATEGIES = [
    {'strategy': 'formation', 'budget': 500},
    {'strategy': 'balance', 'budget': 1000}
]
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')

def load_resources():
    print("Loading resources...")
    from modules.data_loader import load_results, load_payouts
    
    # Load 2025 results
    results = load_results(2025, 2025)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
    
    if 'date' not in results.columns:
        try:
            results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except:
            results['date'] = pd.Timestamp('2024-01-01')

    results['year'] = pd.to_datetime(results['date'], errors='coerce').dt.year
    df_target = results[results['year'] == 2025].copy()
    if 'date' in df_target.columns:
        df_target = df_target.sort_values('date')
        
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    del results
    
    # Load HR
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
        
    # Load Peds
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    # Load Payouts
    returns = load_payouts(2025, 2025)
    
    # Load Model
    model = HorseRaceModel()
    try:
        model.load(MODEL_PATH)
    except:
        print("Model load failed")
        return None

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
    # (Same as simulate_strategy_comparison.py - simplified for brevity in this tool call context, 
    # but in actual file this acts as the processor)
    # Copy-pasting logic from simulate_strategy_comparison.py
    df = race_row_df.copy()
    rename_map = {'枠 番': '枠番', '馬 番': '馬番'}
    df = df.rename(columns=rename_map)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce').fillna(pd.Timestamp('2025-01-01'))
    else:
        df['date'] = pd.Timestamp('2025-01-01')

    for col in ['馬番', '枠番', '単勝']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    try:
        df_proc = predictor.processor.process_results(df)
        if 'horse_id' in df.columns:
            race_hr = hr[hr.index.isin(df['horse_id'].unique())]
        else: race_hr = hr
        
        df_proc = predictor.engineer.add_horse_history_features(df_proc, race_hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, race_hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names: X[c] = df_proc[c] if c in df_proc.columns else 0
        
        # Categorical
        try:
            debug_info = predictor.model.debug_info()
            model_cats = debug_info.get('pandas_categorical', [])
            if len(model_cats) >= 2:
                if '枠番' in X.columns: X['枠番'] = X['枠番'].astype(pd.CategoricalDtype(categories=model_cats[0], ordered=False))
                if '馬番' in X.columns: X['馬番'] = X['馬番'].astype(pd.CategoricalDtype(categories=model_cats[1], ordered=False))
        except: pass

        for c in X.columns:
            if X[c].dtype == 'object' and not isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

        probs = predictor.model.predict(X)
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        if '単勝' in df_res.columns: df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(10.0)
        else: df_res['odds'] = 10.0
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        return df_res
    except Exception as e:
        return None

# Import verify_hit logic (or copy it)
# To avoid dependency issues, copying the verify_hit function logic here
BET_TYPE_MAP = {
    '単勝': 'tan', '複勝': 'fuku', '枠連': 'wakuren', '馬連': 'umaren',
    'ワイド': 'wide', '馬単': 'umatan', '3連複': 'sanrenpuku', '3連単': 'sanrentan'
}
def verify_hit(race_id, rec, payouts_dict):
    if race_id not in payouts_dict: return 0
    race_pay = payouts_dict[race_id]
    payout = 0
    bet_type_jp = rec['bet_type']
    bet_key = BET_TYPE_MAP.get(bet_type_jp)
    if not bet_key or bet_key not in race_pay: return 0
    winning_data = race_pay[bet_key]
    method = rec.get('method', 'SINGLE')
    bet_horses = rec['horse_numbers']
    import itertools
    if bet_key in ['tan', 'fuku']:
        for h in bet_horses:
            if h in winning_data: payout += winning_data[h] * (rec.get('unit_amount', 100) / 100)
    else:
        bought_combinations = []
        if method == 'BOX':
            if bet_key in ['umaren', 'wide']: bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 2)]
            elif bet_key in ['sanrenpuku']: bought_combinations = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 3)]
        elif method in ['FORMATION', '流し']:
            structure = rec.get('formation', [])
            if not structure: structure = [bet_horses]
            if bet_key in ['umaren', 'wide', 'umatan']:
                if len(structure) >= 2:
                    g1, g2 = structure[0], structure[1]
                    for h1 in g1:
                        for h2 in g2:
                            if h1 == h2: continue
                            if bet_key == 'umatan': bought_combinations.append((h1, h2))
                            else: bought_combinations.append(tuple(sorted((h1, h2))))
            elif bet_key in ['sanrenpuku', 'sanrentan']:
                 if len(structure) >= 3:
                     g1, g2, g3 = structure[0], structure[1], structure[2]
                     for h1 in g1:
                        for h2 in g2:
                            if h1 == h2: continue
                            for h3 in g3:
                                if h3 == h1 or h3 == h2: continue
                                if bet_key == 'sanrentan': bought_combinations.append((h1, h2, h3))
                                else: bought_combinations.append(tuple(sorted((h1, h2, h3))))
        elif method == 'SINGLE':
            if bet_key in ['umaren', 'wide', 'sanrenpuku']: bought_combinations.append(tuple(sorted(bet_horses)))
            else: bought_combinations.append(tuple(bet_horses))
        
        bought_combinations = set(bought_combinations)
        for comb in bought_combinations:
            if comb in winning_data: payout += winning_data[comb] * (rec.get('unit_amount', 100) / 100)
    return int(payout)

def main():
    print("=== Exporting Simulation Details ===")
    
    predictor, hr, peds, df_target, returns = load_resources()
    if predictor is None: return

    race_ids = df_target['race_id'].unique()
    
    detail_records = []
    
    print("Running simulation and exporting...")
    for race_id in tqdm(race_ids):
        race_rows = df_target[df_target['race_id'] == race_id]
        if len(race_rows) < 6: continue
        
        # Get basic race info
        date = race_rows.iloc[0]['date']
        venue_code = race_id[4:6] # Simple extraction
        
        df_preds = process_race_data(race_rows, predictor, hr, peds)
        if df_preds is None or df_preds.empty: continue
            
        for setting in TARGET_STRATEGIES:
            strat = setting['strategy']
            bud = setting['budget']
            
            try:
                recs = BettingAllocator.allocate_budget(df_preds, bud, strategy=strat)
            except:
                continue
            
            if not recs:
                # No bet record
                detail_records.append({
                    'race_id': race_id, 'date': date, 'venue_code': venue_code,
                    'strategy': strat, 'budget': bud,
                    'bet_type': 'NONE', 'cost': 0, 'return': 0, 'hit': 0, 'profit': 0
                })
                continue
            
            total_cost = 0
            total_return = 0
            
            # Aggregate per race/strategy
            for rec in recs:
                cost = rec.get('total_amount', rec.get('amount', 0))
                pay = verify_hit(race_id, rec, returns)
                
                total_cost += cost
                total_return += pay
                
            detail_records.append({
                'race_id': race_id, 'date': date, 'venue_code': venue_code,
                'strategy': strat, 'budget': bud,
                'bet_type': 'MIXED', # Simplified
                'cost': total_cost, 'return': total_return,
                'hit': 1 if total_return > 0 else 0,
                'profit': total_return - total_cost
            })
            
    # Save to CSV
    df_out = pd.DataFrame(detail_records)
    output_path = os.path.join(os.path.dirname(__file__), '../../reports/simulation_details_2025.csv')
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    main()
