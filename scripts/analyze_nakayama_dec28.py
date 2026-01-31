
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.constants import RAW_DATA_DIR

def main():
    log_file = 'simulation_2025_log.csv'
    results_file = os.path.join(RAW_DATA_DIR, 'results.pickle')
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
        
    print("Loading data...")
    df_log = pd.read_csv(log_file, dtype={'race_id': str})
    
    # Target Day Prefix (Nakayama 2025 Dec 28)
    # Based on previous analysis: 2025060508
    target_prefix = '2025060508'
    
    print(f"Targeting Day: {target_prefix} (2025/12/28 Nakayama)")
    
    # Filter logs
    day_bets = df_log[df_log['race_id'].astype(str).str.startswith(target_prefix)]
    
    if day_bets.empty:
        print("No bets found for this day.")
        # Try to search widely if prefix is wrong? 
        # But we are sure from previous step.
        return

    # Group by Race ID
    unique_races = sorted(day_bets['race_id'].unique())
    
    # Load Results for Names (Optional but nice)
    race_names = {}
    if os.path.exists(results_file):
        try:
            results = pd.read_pickle(results_file)
            # Try to get race names
            # Fix column name issue
            r_col = None
            for c in ['レース名', 'レース 名']:
                if c in results.columns:
                    r_col = c
                    break
            
            if r_col:
                target_races = results[results.index.astype(str).str.startswith(target_prefix)]
                for rid, row in target_races.iterrows():
                    race_names[str(rid)] = row[r_col]
        except Exception as e:
            print(f"Warning: Could not load race names: {e}")

    print("\n[Results by Race]")
    
    total_inv = 0
    total_ret = 0
    
    for rid in unique_races:
        r_bets = day_bets[day_bets['race_id'] == rid]
        
        inv = r_bets['amount'].sum()
        ret = r_bets['payout'].sum()
        prof = ret - inv
        roi = (ret / inv * 100) if inv > 0 else 0
        
        race_num = str(rid)[-2:]
        r_name = race_names.get(str(rid), "Unknown")
        
        # Format Hits
        hits = r_bets[r_bets['is_hit'] == 1]
        hit_details = []
        if not hits.empty:
            for _, h in hits.iterrows():
                hit_details.append(f"{h['bet_type']}:{h['payout']}")
        
        print(f"R{race_num} {r_name}: Inv:{inv} -> Ret:{ret} ({roi:.0f}%) | Hits: {', '.join(hit_details)}")
        
        total_inv += inv
        total_ret += ret
        
    print("-" * 40)
    grand_prof = total_ret - total_inv
    grand_roi = (total_ret / total_inv * 100) if total_inv > 0 else 0
    
    print(f"TOTAL: Inv:{total_inv:,} -> Ret:{total_ret:,} (+{grand_prof:,}) ROI:{grand_roi:.1f}%")

if __name__ == "__main__":
    main()
