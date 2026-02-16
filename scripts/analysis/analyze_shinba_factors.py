
import pandas as pd
import pickle
import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, MODEL_DIR, PLACE_DICT
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator

def run():
    print("Loading resources...")
    VALIDATION_DIR = os.path.join(MODEL_DIR, "validation_2024")
    # Load model and components
    model = HorseRaceModel()
    model.load(os.path.join(VALIDATION_DIR, 'model.pkl'))
    with open(os.path.join(VALIDATION_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'bias_map.pkl'), 'rb') as f: bias_map = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'jockey_stats.pkl'), 'rb') as f: jockey_stats = pickle.load(f)
    
    # Load results
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
    else:
        results['race_id'] = results.index.astype(str)
        
    # Filter for 2025
    results = results[results['race_id'].str.startswith('2025')].copy()
    
    # Identify Shinba races
    # We need to map race identifiers to "Shinba".
    # Since we can't easily parse race names from `results` (it lacks the name column usually, wait, check preprocessing),
    # `process_results` doesn't extract race name.
    # BUT we know `analyze_2025_breakdown.py` created `race_meta` from `horse_results`.
    # We should replicate that logic or load the date_map if it helps? No, we need the race name.
    # We can load `horse_results` and map race_name to race_id just like `analyze_2025_breakdown`.
    
    print("Mapping Shinba races...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    
    # Load date map to link race_id -> date/venue/R
    with open(os.path.join(os.path.dirname(RAW_DATA_DIR), "date_map_2025.pickle"), 'rb') as f:
        rid_map = pickle.load(f)
        
    # Create simple map: race_id -> is_shinba
    # 1. Parse HR for Shinba races (Date, Venue, R)
    shinba_keys = set()
    hr.columns = hr.columns.str.replace(' ', '')
    if 'レース名' in hr.columns:
        import unicodedata, re
        for _, row in hr.iterrows():
            try:
                name_norm = unicodedata.normalize('NFKC', str(row['レース名']))
                if '新馬' in name_norm:
                    # Parse key
                    match = re.search(r'(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\d+)R', name_norm)
                    if match:
                        date_short, v_info, r_num = match.groups()
                        year_prefix = "20" if int(date_short[:2]) < 50 else "19"
                        date_full = f"{year_prefix}{date_short.replace('/', '-')}"
                        v_match = re.search(r'([^\d]+)', v_info)
                        if v_match:
                            v_name = v_match.group(1).replace('回', '').replace('日', '').strip()
                            shinba_keys.add((date_full, v_name, int(r_num)))
            except: continue
            
    # 2. Identify RaceIDs
    shinba_rids = []
    for rid, date in rid_map.items():
        v_id = rid[4:6]
        v_name = PLACE_DICT.get(v_id)
        if not v_name: continue
        try:
            r_num = int(rid[-2:])
            if (date, v_name, r_num) in shinba_keys:
                shinba_rids.append(rid)
        except: continue
        
    print(f"Found {len(shinba_rids)} Shinba races.")
    if len(shinba_rids) == 0:
        print("No Shinba races found. Check mapping logic.")
        return

    # Process Shinba races
    df_shinba = results[results['race_id'].isin(shinba_rids)].copy()
    if df_shinba.empty:
        print("No results for Shinba IDs.")
        return
        
    df_shinba.set_index('race_id', inplace=True)
    
    # Preprocess
    print("Preprocessing...")
    df_proc = processor.process_results(df_shinba)
    df_proc.index = df_proc.index.astype(str)
    
    # Feature Engineering (Fast path)
    # Pseudo date
    rid_to_date = {rid: pd.Timestamp("2025-01-01") for rid in df_proc.index.unique()}
    df_proc['date'] = df_proc.index.map(rid_to_date)
    
    # Add features
    print("Engineering features...")
    # History (Will be defaults)
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    
    # Need other files?
    with open(os.path.join(RAW_DATA_DIR, "peds.pickle"), 'rb') as f: peds = pickle.load(f)
    
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc = engineer.add_jockey_features(df_proc, jockey_stats)[0]
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc) # Crucial
    df_proc = engineer.add_bias_features(df_proc, bias_map)
    df_proc = processor.encode_categorical(df_proc, ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam'])
    
    # Predict
    print("Predicting...")
    X = df_proc[model.feature_names].fillna(0).copy()
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
    df_proc['probability'] = model.predict(X.fillna(0))
    df_proc['expected_value'] = df_proc['probability'] * df_proc.get('単勝', 0)
    
    # Analyze Strategy for Hybrid 1000
    print("Analyzing Strategy...")
    group_col = 'original_race_id' if 'original_race_id' in df_proc.columns else 'race_id'
    if group_col not in df_proc.columns and df_proc.index.name != 'race_id':
         df_proc = df_proc.reset_index() # Force index to column if needed
         
    # Group by race
    if group_col in df_proc.columns:
        grouper = df_proc.groupby(group_col)
    else:
        grouper = df_proc.groupby(level=0)
        
    strategies_count = {'sanrenpuku_nagashi': 0, 'wide_box': 0, 'other': 0}
    axis_odds = []
    win_horse_probs = []
    
    for rid, race_df in grouper:
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'horse_name': str(row.get('馬名', '')),
                'probability': float(row['probability']),
                'odds': float(row.get('odds', 0) or row.get('単勝', 0)), # Use engineered 'odds' or raw
                'expected_value': float(row['expected_value'])
            })
        df_preds = pd.DataFrame(preds)
        
        # Allocate
        recs = BettingAllocator.allocate_budget(df_preds, 1000, strategy='hybrid_1000')
        if not recs: continue
        
        # Analyze first rec (Main Bet)
        main = recs[0]
        if main['bet_type'] == '3連複' and main['method'] == '流し':
            strategies_count['sanrenpuku_nagashi'] += 1
            # Check Axis Odds
            axis_num = main['horse_numbers'][0] # Axis is first
            axis_row = df_preds[df_preds['horse_number'] == axis_num]
            if not axis_row.empty:
                axis_odds.append(axis_row.iloc[0]['odds'])
        elif main['bet_type'] == 'ワイド' and main['method'] == 'BOX':
             strategies_count['wide_box'] += 1
        else:
             strategies_count['other'] += 1
             
    print("Strategy Breakdown for Shinba:")
    print(strategies_count)
    if axis_odds:
        print(f"Average Axis Odds (Sanrenpuku): {np.mean(axis_odds):.2f}")
        print(f"Median Axis Odds (Sanrenpuku): {np.median(axis_odds):.2f}")

if __name__ == "__main__":
    run()
