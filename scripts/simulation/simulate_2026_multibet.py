"""
2026 YTD 複合馬券シミュレーション
学習済みモデル(production_model.pkl)を使用し、2026/01/01-2026/01/25の期間でシミュレーションを行う。
予算制約: 各券種1レースあたり最大5000円
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import traceback

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel
from modules.strategy_composite import CompositeBettingStrategy

def simulate_2026_multibet():
    try:
        print("=== 2026 YTD Multi-type Betting Simulation (Budget: 5000 Yen/Race/Type) ===")
        
        # 1. Model Loading
        model_path = os.path.join(MODEL_DIR, 'production_model.pkl')
        print(f"[1] Loading Production Model: {model_path}")
        if not os.path.exists(model_path):
            print("Model not found.")
            return
            
        # Load using class method to ensure correct object structure
        model = HorseRaceModel()
        model.load(model_path)
            
        # 2. Data Loading & Preprocessing
        print("[2] Loading 2026 Data...")
        results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
        hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
        peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
        
        results_df = pd.read_pickle(results_path)
        hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
        peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
        
        # Prepare Data
        ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
        
        if len(ret) >= 7:
            X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
        elif len(ret) >= 6:
            X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
            df_full = results_df
        else:
            vals = ret
            X, y = vals[0], vals[1]
            df_full = results_df

        # Extract Year and Date
        if df_full is not None and 'date' in df_full.columns:
            years = df_full['date'].dt.year.values
            dates = df_full['date'].values
            if '着順' in df_full.columns:
                ranks = pd.to_numeric(df_full['着順'], errors='coerce').fillna(0).values
            else:
                ranks = np.zeros(len(df_full))
            
            # Odds
            if 'odds' in df_full.columns:
                odds_arr = df_full['odds'].values
            elif '単勝' in df_full.columns:
                odds_arr = pd.to_numeric(df_full['単勝'], errors='coerce').fillna(0).values
            else:
                print("Odds missing.")
                return
                
            # Horse Number
            if '馬番' in df_full.columns:
                hn_arr = pd.to_numeric(df_full['馬番'], errors='coerce').fillna(0).astype(int).values
            else:
                hn_arr = np.zeros(len(df_full), dtype=int)
        else:
            print("Date column missing.")
            return

        # Create Master DF
        sim_master_df = X.copy()
        
        # Do NOT restore index here to keep RangeIndex for filtering alignment
        # if 'original_race_id' in sim_master_df.columns:
        #    sim_master_df.index = sim_master_df['original_race_id']
                
        sim_master_df['target'] = y
        sim_master_df['year'] = years
        sim_master_df['date'] = dates
        sim_master_df['rank_res'] = ranks
        sim_master_df['odds'] = odds_arr
        sim_master_df['horse_number'] = hn_arr
        
        # Filter 2026/01/01 - 2026/01/25
        start_date = pd.Timestamp('2026-01-01')
        end_date = pd.Timestamp('2026-01-25')
        
        # Ensure date is datetime
        sim_master_df['date'] = pd.to_datetime(sim_master_df['date'])
        
        test_mask = (sim_master_df['date'] >= start_date) & (sim_master_df['date'] <= end_date)
        
        X_test = sim_master_df.loc[test_mask, model.feature_names] 
    
        if len(X_test) == 0:
            print("No data found for 2026/01/01-01/25.")
            return
            
        print(f"Target Races: {len(X_test)} horses in 2026 YTD")
        
        # Categorical conversion
        for col in ['枠番', '馬番']:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('category')
        
        # Predict
        print("[3] Predicting...")
        probs = model.predict(X_test)
        
        # Debug: Score Stats
        print(f"Score Stats: Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}")
        if probs.max() < 0.4:
            print("Warning: Max score is less than 0.4. No bets will be placed.")
        
        # Combine results
        test_df = sim_master_df.loc[test_mask].copy()
        test_df['score'] = probs
        
        # Restore Index from original_race_id column (After filtering)
        if 'original_race_id' in test_df.columns:
            print("Restoring Race ID from original_race_id column (on test_df)...")
            test_df.index = test_df['original_race_id']
        else:
            print("Warning: original_race_id missing in test_df.")
            
        # Debug: Check Index (Race ID)
        print(f"Test DF Index Sample: {test_df.index[:5].tolist()}")

        # Race Partitioning
        if 'race_num' not in test_df.columns:
            rid_str = test_df.index.astype(str)
            test_df['race_num'] = rid_str.str[10:12].astype(int)
            test_df['venue_id'] = rid_str.str[4:6].astype(int)
            
        # Group by unique race identifier
        test_df['race_id_key'] = test_df['date'].astype(str) + '_' + test_df['venue_id'].astype(str) + '_' + test_df['race_num'].astype(str)
        
        print(f"Identified {test_df['race_id_key'].nunique()} races.")
        
        # Load Return Tables
        return_path = 'data/raw/return_tables.pickle'
        print(f"Loading Payouts: {return_path}")
        return_df = pd.read_pickle(return_path)
        print(f"Return Tables Index Sample: {return_df.index[:5].tolist()}")
        
        # Settings
        BOX_SIZES = {
            '単勝': 1,
            '馬連': 5,
            '馬単': 5,
            'ワイド': 5,
            '3連複': 5,
            '3連単': 4
        }
        
        bet_types = ['単勝', '馬連', '馬単', 'ワイド', '3連複', '3連単']
        
        results = {bt: {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0, 'races': 0} for bt in bet_types}
        
        print("Starting Simulation Loop...")
        # Iterate Races
        match_count = 0
        for rid, race_df in tqdm(test_df.groupby('race_id_key')):
            race_id_str = str(race_df.index[0])
            
            payouts = None
            if race_id_str in return_df.index:
                payouts = return_df.loc[race_id_str]
                match_count += 1
            else:
                continue
                
            for bt in bet_types:
                box_n = BOX_SIZES[bt]
                if len(race_df) < box_n: continue
                    
                bets_list = []
                
                if bt == '単勝':
                    top1 = race_df.sort_values('score', ascending=False).iloc[0]
                    if top1['score'] < 0.4: continue
                    bets_list = [{'type': '単勝', 'combo': str(int(top1['horse_number'])), 'amount': 100}]
                else:
                    top1 = race_df.sort_values('score', ascending=False).iloc[0]
                    if top1['score'] < 0.4: continue
                    # Pass correct DF with horse_number
                    bets_list = CompositeBettingStrategy.generate_box_bets(race_df, n_horses=box_n, bet_types=[bt])
                    
                if not bets_list: continue
                    
                invest_sum = sum(b['amount'] for b in bets_list)
                if invest_sum > 5000: continue
                
                ret_sum = CompositeBettingStrategy.calculate_return(bets_list, payouts)
                
                results[bt]['invest'] += invest_sum
                results[bt]['return'] += ret_sum
                results[bt]['races'] += 1
                results[bt]['bets'] += 1
                if ret_sum > 0: results[bt]['hits'] += 1
        
        print(f"Matched {match_count} races with return tables.")
            
        # Report
        print(f"\n=== Simulation Results (2026/01/01 - 2026/01/25) ===")
        print(f"Condition: Top 1 Score >= 0.4 (Strategy B)")
        print(f"{'Type':<8} | {'Box':<4} | {'Races':<6} | {'Invest':<11} | {'Return':<11} | {'Recov %':<8} | {'Hit Rate':<8}")
        print("-" * 80)
        
        summary = []
        for bt in bet_types:
            d = results[bt]
            inv = d['invest']
            ret = d['return']
            recov = (ret / inv * 100) if inv > 0 else 0
            hit_rate = (d['hits'] / d['races'] * 100) if d['races'] > 0 else 0
            
            box_sz = BOX_SIZES[bt]
            print(f"{bt:<8} | {box_sz:<4} | {d['races']:<6} | ¥{inv:<10,} | ¥{ret:<10,} | {recov:>7.1f}% | {hit_rate:>7.1f}%")
            
            summary.append({
                'Type': bt,
                'Box': box_sz,
                'Races': d['races'],
                'Invest': inv,
                'Return': ret,
                'Recovery': recov,
                'HitRate': hit_rate
            })
            
        pd.DataFrame(summary).to_csv('simulation_2026_multibet_summary.csv', index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"CRITICAL ERROR in simulate_2026_multibet: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_2026_multibet()
