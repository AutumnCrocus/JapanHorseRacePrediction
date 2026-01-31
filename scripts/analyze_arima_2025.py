
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
        
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    print("Loading data...")
    df_log = pd.read_csv(log_file, dtype={'race_id': str})
    results = pd.read_pickle(results_file)
    
    # Filter 2025 races
    results_2025 = results[results.index.astype(str).str.startswith('2025')]
    
    # Find Arima Kinen by Logic (Nakayama 11R in late Dec)
    # 2025 Nakayama (06)
    nakayama = results_2025[results_2025.index.astype(str).str[4:6] == '06']
    
    # Sort by ID (descending to find last day)
    nakayama_days = sorted(nakayama.index.astype(str).str[:10].unique(), reverse=True)
    
    if not nakayama_days:
        print("No Nakayama races found in 2025.")
        return
        
    # Assume last day or 2nd last day is Arima Kinen
    target_day = nakayama_days[0] # The very last day
    
    # Arima Kinen is usually 11R
    arima_id = target_day + '11'
    
    print(f"Targeting Arima Kinen ID (Estimated): {arima_id}")
    
    # Verify if exists
    if arima_id not in results_2025.index.astype(str):
        print(f"Warning: {arima_id} not found in results index.")
        # Try to search 10R or other days
    
    # Create pseudo dataframe for iteration
    arima_race = pd.DataFrame({'レース名': ['有馬記念(推定)']}, index=[arima_id])
    
    for rid, row in arima_race.iterrows():
        print(f"Race ID: {rid}, Name: {row['レース名']}")
        
        # Get Date from index or race_id
        # Usually we don't have explicit date column in results.pickle unless processed
        # But we can assume the Day ID (YYYYPPKKDDRR)
        # We want to find ALL races on the SAME DAY as Arima Kinen.
        
        # race_id structure: YYYY(4) Place(2) Kai(2) Day(2) Race(2)
        # Same day means matching YYYYPPKKDD
        
        target_rid = str(rid)
        day_prefix = target_rid[:10] # YYYYPPKKDD
        print(f"Target Day Prefix: {day_prefix}")
        
        # Filter log for this day
        # race_id in log is also full string
        day_bets = df_log[df_log['race_id'].astype(str).str.startswith(day_prefix)]
        
        print(f"\n--- Simulation Results for Arima Kinen Day ({day_prefix}) ---")
        
        # 1. Arima Kinen Specific
        arima_bets = df_log[df_log['race_id'].astype(str) == target_rid]
        print(f"\n[Arima Kinen (Race ID: {target_rid})]")
        if arima_bets.empty:
            print("No bets placed on Arima Kinen.")
        else:
            total_inv = arima_bets['amount'].sum()
            total_ret = arima_bets['payout'].sum()
            print(f"Invest: {total_inv:,} JPY")
            print(f"Return: {total_ret:,} JPY")
            print(f"Profit: {total_ret - total_inv:,} JPY")
            for _, bet in arima_bets.iterrows():
                hit_mark = "🎯" if bet['is_hit'] else "MISS"
                print(f"  {bet['bet_type']} {bet['method']} {bet['combination']}: {bet['amount']} -> {bet['payout']} {hit_mark}")

        # 2. Day Summary
        print(f"\n[Day Total Summary]")
        day_invest = day_bets['amount'].sum()
        day_return = day_bets['payout'].sum()
        day_profit = day_return - day_invest
        day_rate = (day_return / day_invest * 100) if day_invest > 0 else 0
        
        print(f"Total Invest: {day_invest:,} JPY")
        print(f"Total Return: {day_return:,} JPY")
        print(f"Net Profit  : {day_profit:+,} JPY")
        print(f"Recovery Rate: {day_rate:.2f}%")
        print(f"Total Bets  : {len(day_bets)}")
        
        # Show big hits on the day
        big_hits = day_bets[day_bets['payout'] >= 10000]
        if not big_hits.empty:
            print("\n[Big Hits on the Day (>10,000 JPY)]")
            for _, bet in big_hits.iterrows():
                print(f"  Race {bet['race_id']} {bet['bet_type']}: {bet['amount']} -> {bet['payout']:,} JPY")
                
if __name__ == "__main__":
    main()
