import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_prize():
    print("Loading results.pickle...")
    df = pd.read_pickle('data/raw/results.pickle')
    
    # Check data types
    # '賞金 （万円）' might be string with commas or NaN
    # We need to clean it
    print("Cleaning prize data...")
    if df['賞金 （万円）'].dtype == object:
        # Remove commas and convert to float
        # NaN becomes 0 for max calculation purposes (though usually winner has prize)
        df['prize_clean'] = pd.to_numeric(df['賞金 （万円）'], errors='coerce').fillna(0)
    else:
        df['prize_clean'] = df['賞金 （万円）'].fillna(0)
    
    # Group by race_id (index) and take max
    print("Grouping by race...")
    max_prizes = df.groupby(df.index)['prize_clean'].max()
    
    # Filter out 0 prizes (if any)
    max_prizes = max_prizes[max_prizes > 0]
    
    print(f"Total races with prize info: {len(max_prizes)}")
    print("\nPrize Statistics (10k Yen):")
    print(max_prizes.describe())
    
    print("\nPrize Value Counts (Top 30):")
    print(max_prizes.value_counts().sort_index(ascending=False).head(30))
    
    print("\nPotential Grade Thresholds:")
    print("Over 10,000 (G1?):", (max_prizes >= 10000).sum())
    print("Over 5,000 (G2?):", (max_prizes >= 5000).sum())
    print("Over 3,000 (G3?):", (max_prizes >= 3000).sum())
    print("2,500 - 3,000:", ((max_prizes >= 2500) & (max_prizes < 3000)).sum())

if __name__ == "__main__":
    analyze_prize()
