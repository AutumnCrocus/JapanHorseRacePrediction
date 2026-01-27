import pandas as pd
import numpy as np
import os
import sys

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

VENUES = {
    1: '札幌', 2: '函館', 3: '福島', 4: '新潟', 5: '東京', 6: '中山', 7: '中京', 8: '京都', 9: '阪神', 10: '小倉'
}

def recover_details():
    print("=== Recovering Bet Details ===")
    
    # 1. Load Simulation Log
    sim_path = 'rolling_prediction_details.csv'
    if not os.path.exists(sim_path):
        sim_path = 'data/rolling_prediction_details.csv'
    
    if not os.path.exists(sim_path):
        print("Simulation log not found.")
        return

    print(f"Loading simulation log: {sim_path}")
    sim_df = pd.read_csv(sim_path)
    
    # Filter Strategy B (Score >= 0.4)
    # Note: Strategy B logic might verify EV too, but user asked for "Score >= 0.4"
    # Let's filter by Score >= 0.4 first.
    targets = sim_df[sim_df['score'] >= 0.4].copy()
    print(f"Found {len(targets)} bets (Score >= 0.4)")
    
    if len(targets) == 0:
        print("No bets found.")
        return

    # 2. Load Raw Results
    raw_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    print(f"Loading raw results: {raw_path}")
    results_df = pd.read_pickle(raw_path)
    print("Raw Columns:", results_df.columns.tolist()[:20]) # Debug
    
    # Normalize column names
    if '馬 番' in results_df.columns:
        results_df = results_df.rename(columns={'馬 番': '馬番'})
    if '馬番' not in results_df.columns and 'horse_number' in results_df.columns:
        results_df = results_df.rename(columns={'horse_number': '馬番'})
    
    # Preprocess Raw Data to match Simulation Data
    # race_id is usually index
    results_df = results_df.reset_index()
    if 'index' in results_df.columns:
        results_df = results_df.rename(columns={'index': 'race_id'})
    # If index name was race_id, it might be in columns now
    
    # Ensure race_id column exists
    if 'race_id' not in results_df.columns:
        # Check if first column looks like race_id
        first_col = results_df.columns[0]
        if str(results_df[first_col].iloc[0]).isdigit() and len(str(results_df[first_col].iloc[0])) >= 10:
            results_df = results_df.rename(columns={first_col: 'race_id'})
            
    if 'race_id' in results_df.columns:
        results_df['race_id_str'] = results_df['race_id'].astype(str)
        # YYYY PP KK DD RR
        results_df['year'] = results_df['race_id_str'].str[:4].astype(int)
        results_df['venue_id'] = results_df['race_id_str'].str[4:6].astype(int)
        results_df['race_num'] = results_df['race_id_str'].str[10:12].astype(int)
    elif 'date' in results_df.columns:
        results_df['year'] = pd.to_datetime(results_df['date']).dt.year
    else:
        print("Error: Could not extract year info (no race_id or date).")
        print("Columns:", results_df.columns[:10])
        return

    # Create dummy date if missing
    if 'date' not in results_df.columns:
        results_df['date'] = pd.to_datetime(results_df['year'].astype(str) + '-01-01') # Dummy
        
    # Ensure numeric types
    for col in ['venue_id', 'race_num', '着順']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Prepare lookup dictionary for faster access
    # Key: (year, venue_id, race_num) -> DataFrame
    print("Building lookup index...")
    lookup = {}
    grouped = results_df.groupby(['year', 'venue_id', 'race_num'])
    for name, group in grouped:
        lookup[name] = group
        
    # 3. Match and Recover
    print("Matching records...")
    recovered_data = []
    
    # Cache for matched races to avoid re-scanning
    match_count = 0
    fail_count = 0
    
    for idx, row in targets.iterrows():
        key = (int(row['year']), int(row['venue_id']), int(row['race_num']))
        
        candidates = lookup.get(key)
        
        matched_horse = None
        
        if candidates is not None:
            # Try to match by Rank and Odds
            # Note: Odds might have slight float differences, assume close enough
            # Rank should be exact.
            
            # Filter by Rank first
            rank_match = candidates[candidates['着順'] == row['rank']]
            
            if len(rank_match) == 1:
                matched_horse = rank_match.iloc[0]
            elif len(rank_match) > 1:
                # Ambiguous rank (e.g. multiple horses with same rank?), use Odds
                # Find closest odds
                cols_odds = pd.to_numeric(rank_match['単勝'], errors='coerce').fillna(0)
                diff = (cols_odds - row['odds']).abs()
                best_idx = diff.idxmin()
                if diff[best_idx] < 0.2: # Allow small diff
                    matched_horse = rank_match.loc[best_idx]
            else:
                # Rank mismatch (maybe simulation rank logic diff?), try Odds only
                cols_odds = pd.to_numeric(candidates['単勝'], errors='coerce').fillna(0)
                diff = (cols_odds - row['odds']).abs()
                best_idx = diff.idxmin()
                if diff[best_idx] < 0.1:
                    matched_horse = candidates.loc[best_idx]
        
        if matched_horse is not None:
            match_count += 1
            rec = {
                '日付': matched_horse['date'].strftime('%Y-%m-%d'),
                '場所': VENUES.get(int(row['venue_id']), str(int(row['venue_id']))),
                'R': int(row['race_num']),
                '馬番': matched_horse['馬番'],
                '馬名': matched_horse['馬名'],
                'AIスコア': f"{row['score']:.3f}",
                'オッズ': row['odds'],
                '結果': int(row['rank']),
                '券種': '単勝',
                '金額': 100,
                '払戻': int(row['odds'] * 100) if row['rank'] == 1 else 0
            }
            recovered_data.append(rec)
        else:
            fail_count += 1
            # Fallback info
            rec = {
                '日付': f"{int(row['year'])}-??-??",
                '場所': VENUES.get(int(row['venue_id']), str(int(row['venue_id']))),
                'R': int(row['race_num']),
                '馬番': '?',
                '馬名': '不明',
                'AIスコア': f"{row['score']:.3f}",
                'オッズ': row['odds'],
                '結果': int(row['rank']),
                '券種': '単勝',
                '金額': 100,
                '払戻': int(row['odds'] * 100) if row['rank'] == 1 else 0
            }
            recovered_data.append(rec)
            
    print(f"Matched: {match_count}, Failed: {fail_count}")
    
    # 4. Output
    out_df = pd.DataFrame(recovered_data)
    
    # Sort
    out_df = out_df.sort_values(['日付', '場所', 'R', '馬番'])
    
    print("\n--- Strategy B Betting Recovery (First 20) ---")
    print(out_df.head(20).to_string(index=False))
    
    # CSV Save
    out_path = 'data/strategy_b_details_recovered.csv'
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved recovered details to {out_path}")

if __name__ == "__main__":
    recover_details()
