import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_strategy_b_distribution():
    file_path = 'data/strategy_b_details_recovered.csv'
    if not os.path.exists(file_path):
        print("File not found.")
        return

    df = pd.read_csv(file_path)
    
    print(f"Total Bets: {len(df)}")
    
    # 1. Bet Type Distribution
    print("\n[1] Bet Type Distribution")
    type_counts = df['券種'].value_counts(normalize=True) * 100
    print(type_counts)
    
    # 2. Odds Distribution
    print("\n[2] Odds Distribution")
    # bins: <2.0, 2.0-4.9, 5.0-9.9, 10.0-19.9, >=20.0
    bins = [0, 2.0, 5.0, 10.0, 20.0, 1000]
    labels = ['Favorite (<2.0)', 'Middle-High (2.0-4.9)', 'Middle-Low (5.0-9.9)', 'Hole (10.0-19.9)', 'Longshot (>=20.0)']
    df['odds_cat'] = pd.cut(df['オッズ'], bins=bins, labels=labels, right=False)
    
    odds_counts = df['odds_cat'].value_counts(normalize=True) * 100
    print(odds_counts)
    
    # Calculate recovery rate by odds category
    print("\n--- Recovery Rate by Odds Category ---")
    grp = df.groupby('odds_cat', observed=True).apply(lambda x: pd.Series({
        'bets': len(x),
        'hits': len(x[x['結果']==1]),
        'return_rate': x[x['結果']==1]['オッズ'].sum() / len(x) * 100 if len(x)>0 else 0
    }))
    print(grp)
    
    # 3. Venue Distribution
    print("\n[3] Venue Distribution (Top 5)")
    venue_counts = df['場所'].value_counts(normalize=True) * 100
    print(venue_counts.head(5))

if __name__ == "__main__":
    analyze_strategy_b_distribution()
