
import pandas as pd
import numpy as np
import os
import sys

def main():
    log_file = 'simulation_2025_log.csv'
    if not os.path.exists(log_file):
        print(f"File {log_file} not found.")
        return

    df = pd.read_csv(log_file)
    print(f"Loaded {len(df)} bets.")
    
    # Calculate Summary
    total_invest = df['amount'].sum()
    total_payout = df['payout'].sum()
    return_rate = (total_payout / total_invest * 100) if total_invest > 0 else 0
    
    print("-" * 30)
    print(f"Total Invest: {total_invest:,} JPY")
    print(f"Total Payout: {total_payout:,} JPY")
    print(f"Return Rate : {return_rate:.2f}%")
    print(f"Hit Rate    : {df['is_hit'].mean()*100:.2f}%")
    print("-" * 30)
    
    # By Type
    print("\n[By Bet Type]")
    summary_type = df.groupby('bet_type').agg({
        'amount': 'sum',
        'payout': 'sum',
        'is_hit': 'mean'
    })
    summary_type['return_rate'] = summary_type['payout'] / summary_type['amount'] * 100
    print(summary_type)
    
    # By Method
    print("\n[By Method]")
    summary_method = df.groupby('method').agg({
        'amount': 'sum',
        'payout': 'sum',
        'is_hit': 'mean'
    })
    summary_method['return_rate'] = summary_method['payout'] / summary_method['amount'] * 100
    print(summary_method)

if __name__ == "__main__":
    main()
