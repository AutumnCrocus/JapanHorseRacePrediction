import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_graded_failure():
    # 1. Load Data
    sim_path = 'rolling_prediction_details_v2.csv'
    results_path = 'data/raw/results.pickle'
    
    if not os.path.exists(sim_path) or not os.path.exists(results_path):
        print("Data missing.")
        return

    print("Loading data...")
    sim_df = pd.read_csv(sim_path)
    sim_df['original_race_id'] = sim_df['original_race_id'].astype(str)
    
    results_df = pd.read_pickle(results_path)
    
    # 2. Identify Graded Races
    if '賞金 （万円）' in results_df.columns:
        if results_df['賞金 （万円）'].dtype == object:
            results_df['prize_clean'] = pd.to_numeric(results_df['賞金 （万円）'], errors='coerce').fillna(0)
        else:
            results_df['prize_clean'] = results_df['賞金 （万円）'].fillna(0)
            
    max_prizes = results_df.groupby(level=0)['prize_clean'].max()
    graded_ids = max_prizes[max_prizes >= 3000].index.astype(str)
    
    # 3. Split Prediction Data
    sim_df['is_graded'] = sim_df['original_race_id'].isin(graded_ids)
    
    print(f"\nTotal Rows: {len(sim_df)}")
    print(f"Graded Rows: {sim_df['is_graded'].sum()}")
    print(f"Non-Graded Rows: {(~sim_df['is_graded']).sum()}")
    
    # 4. Analyze Top1 Prediction Performance
    # We need to rank horses by score within each race first
    # (Assuming 'score' column exists)
    
    def get_metrics(df, label):
        # Group by race and get Top 1 by score
        # Note: We need to ensure we have the rank info
        # 'rank' column usually contains the actual result.
        
        race_groups = df.groupby('original_race_id')
        
        top1_stats = []
        for rid, group in race_groups:
            # Get horse with max score
            if group.empty: continue
            top_horse = group.loc[group['score'].idxmax()]
            
            top1_stats.append({
                'rank': top_horse['rank'],
                'odds': top_horse['odds'],
                'pop': top_horse['popularity']
            })
            
        metrics_df = pd.DataFrame(top1_stats)
        if metrics_df.empty:
            print(f"[{label}] No data.")
            return

        # Win Rate (Rank 1)
        win_rate = (metrics_df['rank'] == 1).mean()
        # Place Rate (Rank <= 3)
        place_rate = (metrics_df['rank'] <= 3).mean()
        
        # Recover Rate (Simulated for Single Bet)
        # Sum of odds for winners / Total races
        # Assuming flat bet
        winners = metrics_df[metrics_df['rank'] == 1]
        recovery = winners['odds'].sum() / len(metrics_df) * 100
        
        print(f"\n--- {label} ---")
        print(f"Races: {len(metrics_df)}")
        print(f"Model Top1 Win Rate: {win_rate:.2%}")
        print(f"Model Top1 Place Rate: {place_rate:.2%}")
        print(f"Model Top1 Recovery (Flat Bet): {recovery:.2f}%")
        print(f"Avg Odds of Top1 Horse: {metrics_df['odds'].mean():.2f}")
        print(f"Avg Popularity of Top1 Horse: {metrics_df['pop'].mean():.2f}")
        
        # Rank Distribution
        print("Top1 Rank Distribution:")
        print(metrics_df['rank'].value_counts().sort_index().head(5))

    print("\nAnalyzing Non-Graded Races...")
    get_metrics(sim_df[~sim_df['is_graded']], "Non-Graded")
    
    print("\nAnalyzing Graded Races...")
    get_metrics(sim_df[sim_df['is_graded']], "Graded")

if __name__ == "__main__":
    analyze_graded_failure()
