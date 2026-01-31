
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import datetime
import itertools

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.constants import DATA_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, RETURN_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.betting_allocator import BettingAllocator
# from modules.data_loader import DataLoader # Not used and not existing class
from modules.preprocessing import DataProcessor, FeatureEngineer

def load_data():
    print("Loading data...")
    results = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'results.pickle'))
    horse_results = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'horse_results.pickle'))
    return_tables = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'))
    peds = pd.read_pickle(os.path.join(RAW_DATA_DIR, 'peds.pickle'))
    print(f"Data Loaded: Results={len(results)}, HorseResults={len(horse_results)}, ReturnTables={len(return_tables)}")
    return results, horse_results, return_tables, peds

def parse_return_tables(return_tables_df):
    """
    return_tables_df (MultiIndex or DataFrame) -> Dict {race_id: {type: [{'numbers': [1, 2], 'payout': 1000}]}}
    """
    print("Parsing return tables...")
    parsed = {}
    
    # return_tables_df structure check
    # Expecting MultiIndex (race_id, idx) or similar.
    # Columns: 0=type, 1=numbers, 2=payout, 3=popularity
    
    # Reset index to handle easily
    df = return_tables_df.reset_index()
    
    # Column mapping (heuristic based on content)
    # If columns are integers
    type_col = 0
    nums_col = 1
    payout_col = 2
    
    # Check actual column names (might be 'level_0' or 'index' if reset)
    # race_id should be in columns now
    race_id_col = 'level_0' if 'level_0' in df.columns else 'index'
    if race_id_col not in df.columns:
        # Maybe index name was race_id
        if df.columns[0] == 0: # If data only
             # Wait, how to get race_id?
             pass 
    
    # Iterate row by row (slow but safe for verification)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Returns"):
        try:
            # Check where race_id is
            # In reset_index version, race_id should be the first column if original index was race_id
            race_id = str(row[race_id_col]) if race_id_col in row else str(row[0]) # Fallback
            
            # Identify columns for data
            # Assuming standard structural format: Type, Nums, Payout, Pop
             # But columns might be shifted if race_id is 0
            
            # Let's find columns by content type if possible, or assume fixed relative postion
            # return_tables.pickle is usually created by scraping loop:
            # dfs[1] or similar.
            
            # Based on previous output: 0=単勝, 1=9, 2=650円, 3=3人気
            # And after reset_index, they might be named 0, 1, 2, 3
            
            b_type = row[0]
            nums = row[1]
            payout = row[2]
            
            if not isinstance(b_type, str): continue
            
            if race_id not in parsed: parsed[race_id] = {}
            if b_type not in parsed[race_id]: parsed[race_id][b_type] = []
            
            # Clean Payout
            try:
                payout_val = int(str(payout).replace('円', '').replace(',', ''))
            except:
                payout_val = 0
                
            # Clean Numbers
            # '1 - 2' -> [1, 2]
            # '1' -> [1]
            try:
                nums_str = str(nums).replace(' ', '').replace('　', '') # Remove spaces
                # Determine separator
                if '-' in nums_str:
                    n_list = [int(x) for x in nums_str.split('-')]
                elif '→' in nums_str: # Sometimes for exacta/trifecta
                    n_list = [int(x) for x in nums_str.split('→')]
                else:
                    n_list = [int(nums_str)]
            except:
                n_list = []
                
            parsed[race_id][b_type].append({
                'numbers': n_list,
                'payout': payout_val
            })
            
        except Exception as e:
            continue
            
    return parsed

def check_hit(bet_type, numbers, return_data):
    """
    bet_type: '単勝', '3連複' etc.
    numbers: [1, 2] etc.
    return_data: parsed return data for specific race
    
    Returns: Payout Amount (0 if miss)
    """
    if bet_type not in return_data:
        return 0
        
    payout_sum = 0
    
    # Try exact match first
    # For Box/Formation decomposed bets, numbers order matters for exact bets (Trifecta/Exacta)
    # But for Quinella/Trio, order might be flexible in return table? 
    # Usually return table numbers are fixed. But '1-2' vs '2-1'.
    
    # Netkeiba usually lists numbers in order 1-2-3 for Trio, 1->2->3 for Trifecta.
    
    targets = return_data[bet_type]
    
    for t in targets:
        win_nums = t['numbers']
        
        is_hit = False
        
        # Logic per type
        if bet_type in ['単勝', '複勝']:
            if set(numbers) == set(win_nums): is_hit = True
        elif bet_type in ['馬連', 'ワイド', '3連複', '枠連']:
            # Order insensitive
            if set(numbers) == set(win_nums): is_hit = True
        elif bet_type in ['馬単', '3連単']:
            # Order sensitive
            if list(numbers) == list(win_nums): is_hit = True
            
        if is_hit:
            payout_sum += t['payout']
            
    return payout_sum

def main():
    try:
        # 1. Load Data
        results, horse_results, return_tables, peds = load_data()
        
        # Preprocessing to add Date
        results['year'] = results.index.astype(str).str[:4].astype(int)
        results['date'] = pd.to_datetime(results['year'].astype(str) + '-01-01') 
        
        # Split (Simulate separate Data Frames for safety)
        # However, for rolling features we need full context.
        
        train_query = results.index.astype(str).str[:4].astype(int) <= 2024
        test_query = results.index.astype(str).str[:4].astype(int) == 2025
        
        df_train_raw = results[train_query]
        df_test_raw = results[test_query]
        
        print(f"Train Races (<=2024): {len(df_train_raw)}")
        print(f"Test Races (2025): {len(df_test_raw)}")
        
        if len(df_test_raw) == 0:
            print("No test data (2025) found. Aborting.")
            return

        # Parse Returns
        parsed_returns = parse_return_tables(return_tables)

        # 2. Train Model
        print("Preparing Training Data...")
        
        processor = DataProcessor()
        engineer = FeatureEngineer()
        
        full_df = pd.concat([df_train_raw, df_test_raw])
        print(f"DEBUG: Initial full_df: {full_df.shape}")
        
        # Keep race_id as column to survive merges
        # Assuming index is race_id
        full_df.index.name = 'race_id_index' 
        full_df = full_df.reset_index()
        
        # 1. Basic Processing
        # DataProcessor might expect race_id in specific way?
        # processor.process_results usually doesn't change rows, but might reset index?
        # Let's hope it keeps columns.
        
        full_df = processor.process_results(full_df)
        print(f"DEBUG: After process_results: {full_df.shape}")
        sys.stdout.flush()
        
        # 2. Add History
        full_df = engineer.add_horse_history_features(full_df, horse_results)
        print(f"DEBUG: After add_horse_history_features: {full_df.shape}")
        sys.stdout.flush()
        
        # 3. Add Peds
        full_df = engineer.add_pedigree_features(full_df, peds)
        print(f"DEBUG: After add_pedigree_features: {full_df.shape}")
        sys.stdout.flush()
        
        # 4. Add Odds (Clean up)
        full_df = engineer.add_odds_features(full_df)
        print(f"DEBUG: After add_odds_features: {full_df.shape}")
        sys.stdout.flush()
        
        # 5. Encoding
        # encode categorical...
        categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
        categorical_cols = [c for c in categorical_cols if c in full_df.columns]
        full_df = processor.encode_categorical(full_df, categorical_cols)
        print(f"DEBUG: After encode_categorical: {full_df.shape}")

        # Restore Index for Splitting
        if 'race_id_index' in full_df.columns:
            full_df = full_df.set_index('race_id_index')
        elif 'index' in full_df.columns:
            full_df = full_df.set_index('index')
            
        print(f"DEBUG: Restored index. Sample: {full_df.index[:5]}")

        # Split again
        train_mask = full_df.index.isin(df_train_raw.index)
        test_mask = full_df.index.isin(df_test_raw.index)
        
        X_train = full_df[train_mask].copy()
        # Fix Target separation if needed
        if 'rank_num' in full_df.columns:
            y_train = full_df[train_mask]['rank_num'].copy()
        elif '着順' in full_df.columns:
             y_train = pd.to_numeric(full_df[train_mask]['着順'], errors='coerce')
        else:
             print("Error: Target column not found")
             return

        print(f"DEBUG: X_train shape: {X_train.shape}")
        
        # Prepare X, y properly
        
        # Use Whitelist to avoid Leakage (Prize, GoalTime, Odds, etc.)
        feature_whitelist = [
            '枠番', '馬番', '斤量', '年齢',
            '体重', '体重変化', 'course_len',
            'venue_id', 'kai', 'day', 'race_num', # Identifiers/Context
            'avg_rank', 'win_rate', 'place_rate', 'race_count',
            'jockey_avg_rank', 'jockey_win_rate', 'jockey_return_avg',
            'avg_last_3f', 'avg_running_style',
            'interval', 'prev_rank',
            'same_distance_win_rate', 'same_type_win_rate',
            'peds_score_speed', 'peds_score_stamina', 'peds_score_dirt',
            'waku_bias_rate',
            # Categorical (Encoded)
            '性', 'race_type', 'weather', 'ground_state', 'sire', 'dam'
        ]
        
        # Filter existing columns
        use_cols = [c for c in feature_whitelist if c in X_train.columns]
        
        print(f"DEBUG: Selected features ({len(use_cols)}): {use_cols}")
        
        X_train_clean = X_train[use_cols].copy()
        
        print(f"DEBUG: X_train_clean shape: {X_train_clean.shape}")
        sys.stdout.flush()
        
        y_train_binary = (y_train <= 3).astype(int)
        
        # Train
        print("Training Model...")
        model = HorseRaceModel()
        model.train(X_train_clean, y_train_binary)
        print("Model Trained.")

        # 3. Rolling Simulation
        print("Starting Simulation...")
        
        simulation_log = []
        df_test = full_df[test_mask].sort_index()
        
        unique_races = df_test.index.unique()
        
        for rid in tqdm(unique_races, desc="Simulating Races"):
            race_df = df_test.loc[rid]
            # Handle Single row as Series
            if isinstance(race_df, pd.Series):
                race_df = race_df.to_frame().T
            
            # Align features with training data (CRITICAL)
            # Use X_train_clean columns (use_cols)
            # Ensure race_df has same columns in same order
            try:
                # Add missing cols with 0
                missing_cols = set(use_cols) - set(race_df.columns)
                for c in missing_cols:
                    race_df[c] = 0
                
                # Select and Reorder
                X_pred = race_df[use_cols]
                
                probs = model.predict(X_pred)
            except Exception as e:
                print(f"Prediction Error at {rid}: {e}")
                continue
            
            # Create df_preds for allocator
            # df_preds needs '馬番', '馬名' etc which might not be in use_cols
            # So we use race_df (original) for df_preds base, but X_pred for predict
            df_preds = race_df.copy()
            df_preds['probability'] = probs
            
            # Ensure required columns
            # Allocator needs: probability, horse_number, expected_value, odds (optional)
            if '馬番' in df_preds.columns:
                 df_preds['horse_number'] = pd.to_numeric(df_preds['馬番'], errors='coerce').fillna(0).astype(int)
            elif 'horse_number' in df_preds.columns:
                 pass
            else:
                 df_preds['horse_number'] = range(1, len(df_preds)+1)

            # Ensure Odds Column (for allocator logic, though we pass odds_data=None)
            df_preds['odds'] = None
            df_preds['expected_value'] = 0
            
            # Ensure Horse Name
            if '馬名' in df_preds.columns:
                df_preds['horse_name'] = df_preds['馬名']
            elif 'horse_name' in df_preds.columns:
                 pass
            else:
                 df_preds['horse_name'] = [f"H{i}" for i in df_preds['horse_number']]

            budget = 5000
            
            try:
                recommendations = BettingAllocator.allocate_budget(df_preds, budget, odds_data=None)
            except Exception as e:
                print(f"Allocator Error at {rid}: {e}")
                recommendations = []
            
            if not recommendations:
                # Debug print if no recommendations despite budget
                # print(f"DEBUG: No Recs for {rid}. Probs: {probs[:3]}")
                continue
            
            race_return_data = parsed_returns.get(str(rid), {})
            
            for rec in recommendations:
                try:
                    # Decompose BOX/NAGASHI if needed
                    # BettingAllocator returns 'method': 'BOX' etc.
                    # We need to list individual bets for logging/hit check
                    
                    # Fix Keys based on BettingAllocator._format_recommendations
                    bet_type = rec['bet_type']
                    method = rec['method']
                    rec_horses = rec['horse_numbers'] # Was 'horses'
                    amount = rec['total_amount']      # Was 'amount'
                    points = rec['points']            # Was 'count'
                    
                    unit_amount = amount // points if points > 0 else 0
                    
                    # Generate Combinations
                    combos = []
                    h_list = [int(h) for h in rec_horses] # Prepare here
                    
                    if method == 'SINGLE':
                        combos = [tuple(rec_horses)]
                    elif method == 'BOX':
                        if bet_type == '馬連':
                            combos = list(itertools.combinations(h_list, 2))
                        elif bet_type == 'ワイド':
                            combos = list(itertools.combinations(h_list, 2))
                        elif bet_type == '枠連':
                             # Wakuren Box is tricky if horses are used. Need Waku.
                             # If rec['horses'] are Horse Numbers, we need Waku map.
                             # Simplification: Skip Waku Box breakdown or assume Waku numbers in rec['horses']?
                             # Allocator logic returns Horse Numbers.
                             # So Wakuren Box is logically weird in current Allocator unless mapped.
                             combos = [] 
                        elif bet_type == '馬単':
                             combos = list(itertools.permutations(h_list, 2))
                        elif bet_type == '3連複':
                             combos = list(itertools.combinations(h_list, 3))
                        elif bet_type == '3連単':
                             combos = list(itertools.permutations(h_list, 3))
                        else:
                             # Default fallback
                             combos = [(h,) for h in h_list]
                             
                    for c in combos:
                        c_list = list(c)
                        # Check Hit
                        # check_hit expects: bet_type, horses(list)
                        # It returns payout amount
                        
                        # Prepare args for check_hit
                        # check_hit(bet_type, horses, return_data)
                        
                        # Warning: check_hit uses 'tan', 'fuku' codes?
                        # No, parsed_returns keys are 'win', 'place', 'quinella'...
                        # check_hit function usage needs verification.
                        # Let's verify check_hit signature below.
                        
                        hit_amt = check_hit(bet_type, c_list, race_return_data)
                        payout = hit_amt * (unit_amount / 100)
                        
                        is_hit = 1 if hit_amt > 0 else 0
                        
                        simulation_log.append({
                            'race_id': rid,
                            'bet_type': bet_type,
                            'method': method,
                            'combination': str(c_list),
                            'amount': unit_amount,
                            'is_hit': is_hit,
                            'payout': int(payout),
                            'return_rate': 0.0 # Calc later
                        })
                        
                except Exception as e:
                    print(f"Error in processing rec {rec}: {e}")
                    continue
            
            # Progress Log
            if len(simulation_log) % 1000 == 0 and len(simulation_log) > 0:
                 print(f"DEBUG: Logged {len(simulation_log)} bets so far...")

        print("Simulation Loop End.")
        print(f"Total Bets Logged: {len(simulation_log)}")
                    
        # Log Saving
        df_log = pd.DataFrame(simulation_log)
        if not df_log.empty:
            df_log.to_csv("simulation_2025_log.csv", index=False)
            print(f"Simulation Complete. Log saved.")
        else:
            print("No bets made.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
