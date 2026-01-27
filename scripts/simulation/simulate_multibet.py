"""
複合馬券シミュレーションスクリプト
AIスコア上位馬のボックス買い戦略を検証する。
"""
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.strategy_composite import CompositeBettingStrategy
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

def simulate_multibet():
    print("=== Multi-type Betting Simulation (2022-2025) ===")
    
    # 1. Load Data
    sim_path = 'rolling_prediction_details.csv'
    if not os.path.exists(sim_path):
        sim_path = 'data/rolling_prediction_details.csv'
    
    if not os.path.exists(sim_path):
        print("Error: rolling_prediction_details.csv not found.")
        return

    print(f"Loading predictions: {sim_path}")
    # raw predictions usually lack race_id, so we need to group by (year, venue_id, race_num)
    sim_df = pd.read_csv(sim_path)
    
    return_path = 'data/raw/return_tables.pickle'
    print(f"Loading return tables: {return_path}")
    if not os.path.exists(return_path):
        print("Error: return_tables.pickle not found.")
        return
        
    return_df = pd.read_pickle(return_path)
    
    # race_id (Index) -> String for matching
    # return_df index might be MultiIndex or simple Index.
    # inspect_return_tables showed it's likely (race_id, index) or just race_id string.
    # We need to construct race_id from sim_df to match return_df index.
    # race_id format: YYYY PP KK DD RR (12 digits)
    # sim_df has: year, venue_id, race_num. Missing: Kai (KK), Day (DD).
    
    # WARNING: rolling_prediction_details.csv does not have full race_id info (Kai, Day).
    # We need to recover race_id map similarly to recover_bet_details.py
    # OR
    # Use raw results to map (year, venue, race) -> race_id
    
    raw_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    print(f"Loading raw results for ID mapping: {raw_path}")
    results_df = pd.read_pickle(raw_path)
    
    # Preprocess Raw Data mapping
    # (year, venue, race) -> race_id
    race_id_map = {}
    
    results_df = results_df.reset_index()
    if 'index' in results_df.columns:
        results_df = results_df.rename(columns={'index': 'race_id'})
        
    if 'race_id' not in results_df.columns:
         # Try to find race_id col
         if len(results_df) > 0:
             first_val = str(results_df.iloc[0, 0])
             if len(first_val) >= 10 and first_val.isdigit():
                 results_df = results_df.rename(columns={results_df.columns[0]: 'race_id'})

    if 'race_id' in results_df.columns:
        # Create key
        results_df['race_id'] = results_df['race_id'].astype(str)
        results_df['year'] = results_df['race_id'].str[:4].astype(int)
        results_df['venue_id'] = results_df['race_id'].str[4:6].astype(int)
        results_df['race_num'] = results_df['race_id'].str[10:12].astype(int)
        
        # Build map: (year, venue, race) -> race_id
        # Note: race_num 1-12 usually unique per venue/day, but multiple days exist.
        # Wait, (Year, Venue, RaceNum) is NOT unique! (e.g. 2022 Tokyo 1R happens many times)
        # We need DATE to distinguish races.
        # But rolling_prediction_details.csv might NOT have date...
        
        # Let's check sim_df columns.
        if 'date' not in sim_df.columns:
            # We are in trouble if we don't have date.
            # But recover_bet_details.py succeeded in recovering data?
            # It used grouping by (year, venue, race) -> results list
            # And then matched by Rank/Odds.
            # We can use the same approach: Find the race_id that matches the horses' rank/odds sequence.
            pass
    
    # Group sim_df by race candidates
    # Since we can't uniquely identify race by (Year, Venue, Race) alone,
    # we need to process chunk by chunk.
    # Group sim_df by these 3 keys -> List of races (days)
    
    # Better approach:
    # 1. Group sim_df by (year, venue, race_num) -> chunks of horses
    # 2. Inside each chunk, we might have multiple races (different days).
    #    We need to split them.
    #    sim_df usually stores races sequentially.
    #    We can detect race boundary when horse_number resets or count exceeds ~18.
    #    OR
    #    Since rolling simulation processes year by year, they are appended sequentially.
    #    We can iterate and group.
    
    # Let's assume sequential correlation.
    
    # To be robust, let's use the matching logic from recover_bet_details.py
    # For each "Simulated Race" (block of horses), find the "Real Race ID".
    
    # 1. Group sim_df into races
    # We can assign a unique 'sim_race_id' based on row continuity of (year, venue, race).
    # Or simply:
    sim_df['group_key'] = sim_df['year'].astype(str) + '-' + sim_df['venue_id'].astype(str) + '-' + sim_df['race_num'].astype(str)
    
    # Detect change in group_key to identify race blocks?
    # No, same group_key appears on different days.
    # But usually sorted by date in csv.
    
    # Let's add explicit separator logic
    sim_df['prev_key'] = sim_df['group_key'].shift(1)
    sim_df['new_race_flag'] = (sim_df['group_key'] != sim_df['prev_key'])
    
    # Detect change in group_key to identify race blocks
    sim_df['prev_key'] = sim_df['group_key'].shift(1)
    sim_df['new_race_flag'] = (sim_df['group_key'] != sim_df['prev_key'])
    
    # Also, inside same key (same year/venue/race), if horse_number decreases (e.g. 16 -> 1), it's a new race.
    # Check if horse_number exists
    if 'horse_number' in sim_df.columns:
        sim_df['prev_horse'] = sim_df['horse_number'].shift(1).fillna(0)
        sim_df['seq_break'] = (sim_df['horse_number'] < sim_df['prev_horse']) & (sim_df['group_key'] == sim_df['prev_key'])
        sim_df['race_partition'] = (sim_df['new_race_flag'] | sim_df['seq_break']).cumsum()
    else:
        # If no horse_number, we can only rely on group_key.
        # This assumes grouped predictions are unique per race.
        # But (Year, Venue, RaceNum) is NOT unique across days (e.g. 1st Race happens every day).
        # We need another way to split days...
        # 
        # Option: Use chunks of size ~10-18 rows? Risky.
        # Option: Since rolling_prediction_details is appended sequentially, 
        # races are contiguous.
        # We can assume that if (Y, V, R) changes, it's new.
        # But if (Y, V, R) is same as previous row (e.g. adjacent rows), it's same race.
        # Wait, if we have multiple races with same (Y, V, R) appearing sequentially...
        # e.g. Day 1 Race 1, then Day 2 Race 1...
        # The key (Y, V, R) would be SAME.
        # So group_key change detection fails 
        # IF Day 2 Race 1 follows Day 1 Race 1 immediately.
        # But usually data is sorted by Date, Venue, Race.
        # So Date 1 Race 1 -> Date 1 Race 2 -> ... -> Date 2 Race 1.
        # Thus (Y, V, R) will change in between (Race 1 -> Race 2).
        # EXCEPT for the transition from Day 1 Race 12 -> Day 2 Race 1.
        # Here R changes 12 -> 1.
        # So group_key will change.
        # The rare case is if we only have Race 1s from all days sorted together.
        # That is unlikely for rolling simulation output.
        # So relying on group_key change is safe enough.
        
        sim_df['race_partition'] = sim_df['new_race_flag'].cumsum()
    
    print(f"Identified {sim_df['race_partition'].max()} races in simulation data.")
    
    # Prepare Lookup for Real Data (for race_id resolution)
    # Key: (year, venue, race) -> List of {race_id, [ranks], [odds]}
    print("Building Raw Data Lookup...")
    raw_lookup = {}
    
    # Optimization: Filter raw results for 2022-2025 only
    results_df = results_df[results_df['year'] >= 2022].copy()
    
    # Convert necessary columns
    results_df['着順'] = pd.to_numeric(results_df['着順'], errors='coerce').fillna(99)
    if '単勝' in results_df.columns:
        results_df['単勝'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(0)
    elif '単 勝' in results_df.columns:
        results_df['単 勝'] = pd.to_numeric(results_df['単 勝'], errors='coerce').fillna(0)
        
    for name, group in tqdm(results_df.groupby(['year', 'venue_id', 'race_num'])):
        # group contains multiple races (days)
        # Sub-group by race_id
        for rid, r_group in group.groupby('race_id'):
            # Store signature for matching
            # Signature: Sorted ranks or odds sum?
            # Creating a lightweight signature
            sig = {
                'race_id': rid,
                'ranks': sorted(r_group['着順'].tolist()),
                'odds_sum': r_group.get('単勝', r_group.get('単 勝', pd.Series([0]))).sum()
            }
            if name not in raw_lookup:
                raw_lookup[name] = []
            raw_lookup[name].append(sig)

    # Simulation Analysis
    results = {
        '馬連': {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0},
        '馬単': {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0},
        'ワイド': {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0},
        '3連複': {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0},
        '3連単': {'invest': 0, 'return': 0, 'hits': 0, 'bets': 0}
    }
    
    # Strategy Config
    BOX_SIZE = 5 # 3連単は5頭BOXだと60点。100円で6000円。
    
    print(f"Simulating Box {BOX_SIZE} Strategy...")
    
    bet_types = ['馬連', '馬単', 'ワイド', '3連複', '3連単']
    
    valid_race_count = 0
    match_fail_count = 0
    
    # Iterate over simulated races
    grouped_sim = sim_df.groupby('race_partition')
    
    for _, race_df in tqdm(grouped_sim, total=sim_df['race_partition'].max()):
        if len(race_df) < BOX_SIZE:
             continue
             
        # Generate Bets (Score Top N) - MOVED to after horse number resolution
        # bets = CompositeBettingStrategy.generate_box_bets(race_df, n_horses=BOX_SIZE, bet_types=bet_types)
        
        # if not bets:
        #    continue
            
        # Resolve Real Race ID
        # Metadata
        current_year = int(race_df.iloc[0]['year'])
        current_venue = int(race_df.iloc[0]['venue_id'])
        current_race = int(race_df.iloc[0]['race_num'])
        
        key = (current_year, current_venue, current_race)
        candidates = raw_lookup.get(key)
        
        matched_rid = None
        
        if candidates:
            # Match by rank sequence similarity
            # Sim ranks
            current_ranks = sorted(race_df['rank'].tolist())
            current_odds_sum = race_df['odds'].sum()
            
            best_score = float('inf')
            
            for cand in candidates:
                # Compare odds sum (easy & fast)
                diff = abs(cand['odds_sum'] - current_odds_sum)
                if diff < best_score:
                    best_score = diff
                    matched_rid = cand['race_id']
                
                # If perfect match on ranks (if data is complete)
                # But sim data might be filtered/incomplete?
                # Assuming sim data has all horses?
                # Usually yes.
                
            # If difference is too large, it might be mismatch
            if best_score > 100.0: # Arbitrary threshold
                matched_rid = None
        
        if not matched_rid:
            match_fail_count += 1
            continue
            
        valid_race_count += 1
        
        # Get Return Table & Horse Info
        try:
            # Payouts
            if matched_rid in return_df.index:
                race_payouts = return_df.loc[matched_rid]
            else:
                continue
                
            # Horse Info (from raw results block)
            # Find the raw data block for this race_id
            # raw_lookup stores list of dicts. We need the full dataframe slice.
            # optimized lookup only stored signature. We need to fetch from results_df.
            
            # Since results_df is huge, filtering by race_id each time is slow.
            # But we filtered results_df to 2022+ already.
            # Let's index results_df by race_id for speed.
        except KeyError:
            continue
            
    # Pre-indexing results for fast access
    print("Indexing raw results by race_id...")
    results_map = {rid: grp for rid, grp in results_df.groupby('race_id')}
    
    print(f"Simulating Box {BOX_SIZE} Strategy...")
    
    bet_types = ['馬連', '馬単', 'ワイド', '3連複', '3連単']
    
    valid_race_count = 0
    match_fail_count = 0
    
    # Iterate over simulated races
    grouped_sim = sim_df.groupby('race_partition')
    
    for _, race_df in tqdm(grouped_sim, total=sim_df['race_partition'].max()):
        if len(race_df) < BOX_SIZE:
             continue
             
        # Resolve Real Race ID first to get horse numbers
        current_year = int(race_df.iloc[0]['year'])
        current_venue = int(race_df.iloc[0]['venue_id'])
        current_race = int(race_df.iloc[0]['race_num'])
        
        key = (current_year, current_venue, current_race)
        candidates = raw_lookup.get(key)
        
        matched_rid = None
        
        if candidates:
            current_ranks = sorted(race_df['rank'].tolist())
            current_odds_sum = race_df['odds'].sum()
            
            best_score = float('inf')
            for cand in candidates:
                diff = abs(cand['odds_sum'] - current_odds_sum)
                if diff < best_score:
                    best_score = diff
                    matched_rid = cand['race_id']
                
            if best_score > 100.0:
                matched_rid = None
        
        if not matched_rid:
            match_fail_count += 1
            continue
            
        # Get Real Horse Data (Map Rank -> Horse Number)
        if matched_rid not in results_map:
            continue
            
        real_race_df = results_map[matched_rid]
        
        # VALIDATION: Check if ranks match strictly
        # sim_df ranks vs real_race_df ranks
        real_ranks = sorted(real_race_df['着順'].tolist())
        current_ranks = sorted(race_df['rank'].tolist())
        
        # Length check
        if len(real_ranks) != len(current_ranks):
            # print(f"DEBUG: Rank length mismatch. Sim: {len(current_ranks)}, Real: {len(real_ranks)}")
            match_fail_count += 1
            continue
            
        # Value check (allow small mismatch if cancelled horses?)
        # Exact match required for safety
        if real_ranks != current_ranks:
            # print(f"DEBUG: Rank content mismatch for {matched_rid}")
            match_fail_count += 1
            continue
            
        # If passed, we trust this match
        
        # We need to assign `horse_number` to `race_df` based on matching.
        # Simple match by `rank`. (Assuming rank is unique enough or consistent)
        # If ranks are duplicated (e.g. DNF=99), we might have issues.
        # But 'rank' in sim_df comes from raw data originally, so it should match.
        
        # Create map: rank -> horse_num
        # Handle duplicate ranks? (e.g. tie) -> rare but possible.
        # Use simple map first.
        
        rank_to_umaban = {}
        for _, row in real_race_df.iterrows():
            r = row['着順']
            # Normalize horse num name
            hn = row.get('馬番', row.get('馬 番', 0))
            rank_to_umaban[r] = hn
            
        # Assign horse_number to race_df
        # Using a list comprehension to avoid index alignment issues
        race_df_ranks = race_df['rank'].values
        mapped_horse_nums = []
        for r in race_df_ranks:
            mapped_horse_nums.append(rank_to_umaban.get(r, 0))
            
        # Create a copy to modify
        race_df_bet = race_df.copy()
        race_df_bet['horse_number'] = mapped_horse_nums
        
        # Filter out rows where horse_number is missing (0)
        race_df_bet = race_df_bet[race_df_bet['horse_number'] != 0]
        
        if len(race_df_bet) < BOX_SIZE:
            continue

        valid_race_count += 1
        
        # Generate Bets (Score Top N)
        # Now race_df_bet has 'score' and 'horse_number'
        # Debug
        if 'horse_number' not in race_df_bet.columns:
            print(f"Error: horse_number missing columns: {race_df_bet.columns}")
            continue
            
        bets = CompositeBettingStrategy.generate_box_bets(race_df_bet, n_horses=BOX_SIZE, bet_types=bet_types)
        
        if not bets:
            continue
            
        # Get Return Table
        try:
            if matched_rid in return_df.index:
                race_payouts = return_df.loc[matched_rid]
            else:
                continue
        except KeyError:
            continue
            
        # Calculate Returns
        # Group bets by type
        r_invest = {t: 0 for t in bet_types}
        r_bets = {t: [] for t in bet_types}
        
        for b in bets:
            r_invest[b['type']] += b['amount']
            r_bets[b['type']].append(b)
            
        # Check hits
        for b_type in bet_types:
            type_bets = r_bets[b_type]
            if not type_bets: continue
            
            ret_amt = CompositeBettingStrategy.calculate_return(type_bets, race_payouts)
            
            results[b_type]['invest'] += r_invest[b_type]
            results[b_type]['return'] += ret_amt
            results[b_type]['bets'] += len(type_bets)
            if ret_amt > 0:
                results[b_type]['hits'] += 1
                
    # Report
    print(f"\nSimulation Complete.")
    print(f"Valid Races: {valid_race_count} (Matches), Failed: {match_fail_count}")
    
    summary_data = []
    
    print("\n--- Multi-type Betting Results (Box 5) ---")
    print(f"{'Type':<10} | {'Bets':<8} | {'Invest':<12} | {'Return':<12} | {'Recov %':<8} | {'Hit Rate':<8}")
    print("-" * 75)
    
    for b_type in bet_types:
        d = results[b_type]
        invest = d['invest']
        ret = d['return']
        recov = (ret / invest * 100) if invest > 0 else 0
        hit_rate = (d['hits'] / valid_race_count * 100) if valid_race_count > 0 else 0
        
        print(f"{b_type:<10} | {d['bets']:<8} | ¥{invest:<11,} | ¥{ret:<11,} | {recov:>7.1f}% | {hit_rate:>7.1f}%")
        
        summary_data.append({
            'Type': b_type,
            'Strategy': f'Box {BOX_SIZE}',
            'Bets': d['bets'],
            'Invest': invest,
            'Return': ret,
            'Recovery': recov,
            'HitRate': hit_rate
        })

    # CSV Save
    pd.DataFrame(summary_data).to_csv('data/multibet_results.csv', index=False, encoding='utf-8-sig')
    print("\nSaved to data/multibet_results.csv")

if __name__ == "__main__":
    simulate_multibet()
