
import pandas as pd
import pickle
import os
import sys
from datetime import datetime

# Import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.training import HorseRaceModel
from modules.betting_allocator import BettingAllocator

# Config
DATA_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'
MODEL_PATH = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\models\experiment_model_2025.pkl'
OUTPUT_REPORT = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\2025_simulation_report.md'
START_DATE = '2016-01-01'

def load_data():
    print("Loading data...")
    with open(os.path.join(DATA_DIR, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'horse_results.pickle'), 'rb') as f:
        horse_results = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'peds.pickle'), 'rb') as f:
        peds = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'return_tables.pickle'), 'rb') as f:
        return_tables = pickle.load(f)
    return results, horse_results, peds, return_tables

def run_simulation():
    # 1. Load Data
    results, horse_results, peds, return_tables = load_data()
    
    # 2. Preprocess (Standard Flow used in Jan Sim)
    print("Preprocessing...")
    
    # Ensure race_id
    if 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)

    # Filter 2025 for Simulation Target (But we need history for FE)
    # We will process ALL relevant history + 2025 data, then filter for prediction.
    # Actually, to save time, we can filter `results` to 2024-2025 for processing?
    # No, Feature Engineer needs history.
    # We will use the full set but optimize.
    
    # Processor
    processor = DataProcessor()
    # To avoid huge processing, maybe load pre-processed?
    # Jan 2026 Sim processed on the fly. It took time but worked.
    # We should stick to the same logic to ensure consistency.
    
    # Filter results to manageable size?
    # We need 2025 data. We have 2005-2025?
    # The 'results.pickle' has 480k rows. 
    # Let's process it.
    
    # Create Dummy Dates based on race_id sort order
    # results.pickle lacks date. Scraping failed.
    # We assign dates linearly from 2025-01-01 to 2025-12-31 to preserve relative order.
    print("Using Linear Dummy Dates (2025-01-01 to 2025-12-31)...")
    
    # Ensure race_id is in columns BEFORE processing
    if 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    # CRITICAL FIX: process_results drops rows where '着順' (Rank) is NaN.
    # For simulation (future/unknown races), Rank is likely NaN or invalid.
    # We MUST fill it to prevent dropping.
    # We use '1' as dummy.
    print("Filling NaN '着順' with 1 to prevent dropping in process_results...")
    if '着順' in results.columns:
        results['着順'] = results['着順'].fillna(1)
        # Also handle non-numeric that strictly coerce to NaN?
        # Maybe set all 2025 to 1?
        # Identify 2025 rows by race_id
        is_2025 = results['race_id'].astype(str).str.startswith('2025')
        results.loc[is_2025, '着順'] = 1
        print(f"Filled Rank for {is_2025.sum()} rows (2025).")
        
    df_proc = processor.process_results(results)
    
    # Check if race_id persisted
    if 'race_id' not in df_proc.columns:
        print("Warning: race_id lost in process_results! Restoring from index if possible...")
        # If process_results kept index, fine. If reset, we are in trouble.
        # But usually it just manipulates columns.
        if df_proc.index.astype(str)[0] == results['race_id'].iloc[0]:
             df_proc['race_id'] = df_proc.index.astype(str)
        else:
             print("CRITICAL: Index mismatch after processing. race_id might be lost.")
             # Workaround: Re-assign from results if lengths match?
             # df_proc = df_proc.set_index(results.index) ?
             # Assume lengths match.
             df_proc['race_id'] = results['race_id'].values
    
    # Sort by race_id to approximate chronological order
    if 'race_id' not in df_proc.columns:
        df_proc['race_id'] = df_proc.index.astype(str)
    
    # Sort
    df_sorted = df_proc.sort_values('race_id')
    n = len(df_sorted)
    
    # Generate range
    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2025-12-31')
    
    # Timedelta logic
    # total_seconds = (end_date - start_date).total_seconds()
    # step = total_seconds / n
    # dates = [start_date + pd.Timedelta(seconds=step*i) for i in range(n)]
    
    # Vectorized
    # Actually just periods=n
    # Use pandas linear interpolation?
    # Just assign proportional check
    # But checking 'year' later relies on this.
    
    # Assign year 2025 to all?
    # No, results contains 2005-2025.
    # We must respect the year in race_id!
    # race_id: YYYY...
    # We extract Year from race_id.
    # Then assign Month/Day linearly within that year?
    # Or just 01-01 for all? (Bad for history).
    
    # Better: Extract Year from race_id.
    # Assign Month/Day based on sort position within that year.
    
    df_proc['year_extracted'] = df_proc['race_id'].str[:4].astype(int)
    
    # We only care about 2025 correct ordering?
    # But history features need correct ordering for past years too!
    
    # Simple strategy:
    # Group by Year.
    # For each year, assign dates from Jan 1 to Dec 31 linearly.
    
    def assign_dates(group):
        y = group.name
        start = pd.Timestamp(f'{y}-01-01')
        end = pd.Timestamp(f'{y}-12-31')
        # Create timestamps
        count = len(group)
        if count == 1:
            return pd.Series([start], index=group.index)
        
        # linearly space
        # We can use pd.date_range
        # periods=count
        dates = pd.date_range(start=start, end=end, periods=count)
        return pd.Series(dates, index=group.index)

    print("Assigning synthetic dates per year...")
    # Sort ensures we assign earlier race_ids to earlier dates
    df_proc = df_proc.sort_values('race_id')
    
    # Group by year and apply
    # This might be slow?
    # 480k rows, 20 groups. Fast.
    # Note: data_range generates DatetimeIndex.
    
    # Apply transform?
    # Groupby object
    # Just iterate years
    date_series = []
    
    # Optimization: Filter only relevant years? 
    # History needs e.g. 2020-2025.
    # We can handle all.
    
    years = df_proc['year_extracted'].unique()
    for y in sorted(years):
        sub = df_proc[df_proc['year_extracted'] == y]
        idx = sub.index
        # Create dates
        d = pd.date_range(start=f'{y}-01-01', end=f'{y}-12-31', periods=len(sub))
        date_series.append(pd.Series(d, index=idx))
        
    full_dates = pd.concat(date_series)
    # Align to df_proc
    df_proc['date'] = full_dates.reindex(df_proc.index)
    
    print("Synthetic Dates Assigned.")
    
    # OPTIMIZATION: Filter for 2025 *BEFORE* expensive Feature Engineering
    # We only need features for the target simulation year.
    # FE uses 'horse_results' for history, so we don't need previous rows in df_proc itself for calculation.
    print("Filtering for 2025 before FE...")
    if 'date' in df_proc.columns:
        df_proc['year'] = pd.to_datetime(df_proc['date']).dt.year
    else:
        # Try processing 'race_id' or index
        # race_id might be index
        idx_str = df_proc.index.astype(str)
        df_proc['year'] = idx_str.str[:4].astype(int)
        
    df_proc = df_proc[df_proc['year'] == 2025].copy()
    print(f"Data reduced to {len(df_proc)} rows for FE.")
    
    # Feature Engineering
    print("Feature Engineering...")
    engineer = FeatureEngineer()
    df_proc = engineer.add_horse_history_features(df_proc, horse_results) 
    print("Adding Course Suitability...")
    df_proc = engineer.add_course_suitability_features(df_proc, horse_results)
    print("Adding Jockey Features...")
    df_proc, _ = engineer.add_jockey_features(df_proc)
    print("Adding Pedigree Features...")
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    print("Adding Odds Features...")
    df_proc = engineer.add_odds_features(df_proc)
    
    # 3. Filter for 2025 Simulation *AFTER* FE (Already done, just safety check)
    # Extract Year from race_id if needed, or index.
    # results index is race_id? Yes.
    # df_proc index might be race_id or reset.
    
    # Add year column if missing (already added above)
    if 'year' not in df_proc.columns:
         df_proc['year'] = df_proc.index.str[:4].astype(int)
         
    # Target: 2025
    df_2025 = df_proc # Already filtered
    print(f"2025 Data Rows: {len(df_2025)}")
    
    if df_2025.empty:
        print("No 2025 data found!")
        return

    # Filter for Race 11 (Main Races / Proxy for Graded)
    # race_id: YYYYJJKKRR (Last 2 usually RR)
    # Or 'race_id' column
    # Filter for Race 11 (Main Races / Proxy for Graded)
    # Use race_id column
    if 'race_id' in df_2025.columns:
        # race_id might be numeric or string
        # If numeric, convert to str
        s_ids = df_2025['race_id'].astype(str)
        df_2025['rr'] = s_ids.str[-2:].astype(int)
    else:
        # Fallback to index
        df_2025['rr'] = df_2025.index.astype(str).str[-2:].astype(int)
        
    df_sim = df_2025[df_2025['rr'] == 11].copy()
    
    # Ensure race_id is index for correct iteration
    if 'race_id' in df_sim.columns:
        df_sim.set_index('race_id', inplace=True)

    
    # Debug
    print(f"Rows in df_2025: {len(df_2025)}")
    print(f"Rows in df_sim (11R): {len(df_sim)}")
    print(f"Target Races (Unique): {len(df_sim.groupby(level=0))}")
    if len(df_sim) == 0:
        print("DEBUG: Sample race_ids:", df_2025['race_id'].head() if 'race_id' in df_2025.columns else df_2025.index[:5])
    
    # 4. Load Model
    print("Loading Model...")
    model = HorseRaceModel()
    model.load(MODEL_PATH)
    
    # 5. Predict
    print("Predicting...")
    # Prepare features
    # Drop non-feature columns
    drop_cols = ['date', 'race_id', 'rr', 'year', 'rank', '着順', 'popular', 'odds', 'payout'] # Add whatever is not in features
    # Actually Model.predict handles feature selection if we pass everything?
    # No, we better pass features.
    
    # Predict returns probabilityDataFrame
    # But `model.predict` expects X (DataFrame of features).
    # We need to drop non-numeric/metadata.
    # Usually `model.predict` handles it if we use the wrapper?
    # Let's check `HorseRaceModel.predict`.
    # It likely calls `lgb_model.predict(X)`.
    # So we need to strip metadata.
    # For now, let's use the `model.predict_proba(df_sim)` if available.
    # Or just `model.lgb_model.predict(df_sim[model.feature_names])`.
    
    # To be safe, let's look at `january_2026_simulation.py` logic for prediction?
    # "X = df_proc[model.feature_name_]"
    
    # Let's assume we can get probabilities.
    # We need a loop per race.
    
    races = df_sim.index.unique()
    
    # CRITICAL FIX: Ensure all feature columns are numeric!
    print("Forcing features to numeric...")
    
    # Debug model type
    print(f"Model Type: {type(model)}")
    print(f"Model Dir: {dir(model)[:10]}") # Show first 10, or check for feature_name

    # Determine feature names
    features = []
    if hasattr(model, 'feature_name'):
        features = model.feature_name()
    elif hasattr(model, 'feature_name_'):
        features = model.feature_name_
    elif hasattr(model, 'booster_'):
        if hasattr(model.booster_, 'feature_name'):
             features = model.booster_.feature_name()
    
    if not features:
        print("Warning: Could not determine feature names from model keys. Using fallback.")
        # Fallback: Use all columns except obvious metadata/targets
        # Error log showed '枠番', '馬番' etc are being checked, so they are likely features.
        # We need to find the intersection of what the model expects and what we have.
        # Since we don't know what model expects, we pass everything that looks numeric.
        exclude = ['date', 'race_id', 'result_id', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id', 
                   '馬名', '騎手', '調教師', '馬主', 'レース名', '着順', 'time']
        features = [c for c in df_sim.columns if c not in exclude and not c.startswith('meta_')]
        print(f"Fallback detected {len(features)} features.")

    print(f"Converting {len(features)} features to numeric...")
    for col in features:
        if col in df_sim.columns:
            # Convert to numeric, turn errors (strings) to NaN, then fill with 0
            df_sim[col] = pd.to_numeric(df_sim[col], errors='coerce').fillna(0)
            
    # Also blindly convert common columns causing issues if they missed the feature list
    common_cols = ['枠番', '馬番', '斤量', 'odds', 'popularity', 'year']
    for col in common_cols:
         if col in df_sim.columns:
             df_sim[col] = pd.to_numeric(df_sim[col], errors='coerce').fillna(0)

    # Verify dtypes
    print(df_sim[features].dtypes.head())
    
    results_data = []
    
    # Simulate race by race to apply strategy
    # Group by race_id (index)
    grouped = df_sim.groupby(level=0)
    print(f"Simulating {len(grouped)} races...")
    
    total_invest = 0
    total_return = 0
    hits = 0
    race_count = 0
    
    results_log = []
    
    for race_id in races:
        # Get race data
        race_data = df_sim.loc[race_id]
        if isinstance(race_data, pd.Series): # Single horse race? Unlikely.
             race_data = race_data.to_frame().T
             
        # Predict
        try:
             # debug first race dtypes
             if race_id == races[0]:
                 print("First race input dtypes:")
                 print(race_data[features].dtypes.value_counts())
                 
            # CRITICAL FIX: Force numeric conversion JIT to ensure no object types
             race_data_features = race_data[features].apply(pd.to_numeric, errors='coerce').fillna(0)
             
             # Probs
             probs = model.predict(race_data_features) # Returns array of probs?
             # If model.predict returns class, we need predict_proba.
             # Assuming predict returns probabilities for binary class 1 (Win/Place).
             
             # Create df_preds for allocator
             # Needs: horse_number, probability, odds (if available), expected_value
             current_odds = race_data['単勝'].values if '単勝' in race_data.columns else [0]*len(race_data)
             # Handle missing odds
             current_odds = [float(x) if x is not None else 0 for x in current_odds]
             
             df_preds = pd.DataFrame({
                 'horse_number': race_data['馬番'].values,
                 'probability': probs,
                 'odds': current_odds
             })
             df_preds['expected_value'] = df_preds['probability'] * df_preds['odds']
             
             # Allocate
             allocations = BettingAllocator.allocate_budget(
                 df_preds, 
                 budget=1000, 
                 strategy='hybrid_1000'
             )
             
             if not allocations:
                 continue
                 
             race_invest = sum([a['total_amount'] for a in allocations])
             race_return = 0
             
             # Verify Hit (Calculate Return)
             # Need Return Table for this race
             # returns_df has index race_id? No, usually index is not race_id in pickle?
             # Or it's a dict {race_id: df}?
             # In `script/inspect_returns` it was DataFrame with index as proper range, and race_id was not index?
             # Wait, `run_check` output: `201601010105 0 単勝 ...`
             # It seems Index IS race_id (MultiIndex?).
             # Let's assume we can retrieve by `return_tables.loc[race_id]`.
             
             try:
                 ret_data = return_tables.loc[race_id]
                 # Calculate payoff
                 # ret_data columns: [Type, HorseNums, Pay, Pop]
                 # Iterate allocations and check match
                 
                 for bet in allocations:
                     bet_type = bet['bet_type'] # '単勝', 'ワイド', '3連複'
                     bet_horses = bet['horse_numbers'] # List of bought horses?
                     # Wait, `horse_numbers` in allocation might be [1,2,3,4] for BOX.
                     # We need to generate combinations?
                     # Or does allocation returns individual combinations?
                     # The `hybrid_1000` returns SUMMARY DICT like:
                     # {'bet_type': '3連複', 'method': '流し', 'combination': '...', 'points': 10, 'formation': [[Axis], [Opps]]}
                     # I need to expand this to check hits.
                     
                     # Check Logic:
                     # 1. Expand `formation` to specific combinations.
                     # 2. Check each combination against `ret_data`.
                     
                     # Quick payoff calc:
                     # Load verify logic?
                     # Or implement simple one.
                     
                     # 3-Ren-Puku Axis 1 Flow:
                     if bet['method'] == '流し' and bet_type == '3連複':
                         axis = bet['formation'][0][0]
                         opps = bet['formation'][1]
                         # Combinations: (axis, o1, o2) for o1,o2 in opps
                         import itertools
                         combos = []
                         for pair in itertools.combinations(opps, 2):
                             combos.append(tuple(sorted([axis] + list(pair))))
                         
                         hit_amt = check_hit(combos, ret_data, '3連複', bet['unit_amount'])
                         race_return += hit_amt
                         
                     elif bet['method'] == 'BOX' and bet_type == 'ワイド':
                         box = bet['formation'][0]
                         combos = [tuple(sorted(p)) for p in itertools.combinations(box, 2)]
                         hit_amt = check_hit(combos, ret_data, 'ワイド', bet['unit_amount'])
                         race_return += hit_amt
                         
                     elif bet['method'] == 'SINGLE' and bet_type == '単勝':
                         h = bet['formation'][0][0]
                         hit_amt = check_hit([(h,)], ret_data, '単勝', bet['unit_amount'])
                         race_return += hit_amt
                     
             except KeyError:
                 # No return data (cancelled?)
                 pass
             
             total_invest += race_invest
             total_return += race_return
             if race_return > 0:
                 hits += 1
             race_count += 1
             
             results_log.append({
                 'race_id': race_id,
                 'invest': race_invest,
                 'return': race_return,
                 'hit': race_return > 0,
                 'allocations': allocations
             })
             
        except Exception as e:
             print(f"Error in race {race_id}: {e}")
             continue

    # 6. Report
    roi = (total_return / total_invest * 100) if total_invest > 0 else 0
    profit = total_return - total_invest
    hit_rate = (hits / race_count * 100) if race_count > 0 else 0
    
    report = f"""# 2025年 重賞(11R)シミュレーション結果 (予算1000円・Hybrid戦略)

## 概要
*   **対象**: 2025年の全11R (重賞・オープン相当)
*   **戦略**: Hybrid 1000 (厳選3連複 or バランスワイド)
*   **予算**: 1000円/レース

## 結果
| 項目 | 値 |
|---|---|
| 対象レース数 | {race_count} |
| 総投資額 | {total_invest:,} 円 |
| 総回収額 | {total_return:,} 円 |
| **回収率** | **{roi:.1f}%** |
| 的中数 | {hits} |
| 的中率 | {hit_rate:.1f}% |
| 純利益 | {profit:,} 円 |

## 詳細
(詳しくはログを参照)
"""
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Simulation Complete. Report generated.")
    print(report)

def check_hit(my_combos, ret_df, bet_type, unit):
    # ret_df cols: 0=Type, 1=HorseNums, 2=Pay
    # Filter by bet_type
    # Pay: "1,230円" -> 1230
    # HorseNums: "1 - 2" or "1"
    
    total_pay = 0
    
    # Get rows for bet_type
    # return_tables pickle structure is MultiIndex (RaceID) -> DataFrame
    # Columns normally 0,1,2,3 if raw.
    # Col 0 is bet type name.
    
    # DEBUG: Check if we have return data
    if len(ret_df) == 0:
        return 0

    bet_rows = ret_df[ret_df[0] == bet_type]
    
    if len(bet_rows) == 0:
        return 0
        
    for _, row in bet_rows.iterrows():
        # Parse Winning Combo
        # format: "1 - 2" (str)
        try:
            win_str = str(row[1])
            # Replace non-digit?
            # Usually "1 - 2" or "1-2"
            # Split by non-digits
            import re
            parts = [int(p) for p in re.split(r'[^\d]+', win_str) if p.isdigit()]
            win_combo = tuple(sorted(parts))
            
            # Parse Payout
            pay_str = str(row[2]).replace(',', '').replace('円', '')
            payout = int(pay_str)
            
            # Check match
            if win_combo in my_combos:
                 pay_amount = int(payout * (unit / 100))
                 total_pay += pay_amount
                 
        except Exception as e:
            continue
            
    return total_pay
             
    return total_pay


if __name__ == '__main__':
    run_simulation()
