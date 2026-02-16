import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# Reuse functions from simulation script (simplified)
from simulate_strategy_comparison import load_resources, process_race_data

def run_one_race():
    print("Loading resources...")
    predictor, hr, peds, results, returns = load_resources()
    
    # 2025 Data
    results['year'] = pd.to_datetime(results['date'], errors='coerce').dt.year
    df_2025 = results[results['year'] == 2025]
    race_ids = df_2025['race_id'].unique()
    
    if len(race_ids) == 0:
        print("No 2025 races found. Using 2024.")
        df_2025 = results[results['year'] == 2024]
        race_ids = df_2025['race_id'].unique()

    # Pick a random race
    import random
    random.seed(42)
    race_id = random.choice(race_ids)
    print(f"Selected Race: {race_id}")
    
    race_rows = df_2025[df_2025['race_id'] == race_id]
    
    print("Processing Data...")
    df_preds = process_race_data(race_rows, predictor, hr, peds)
    
    if df_preds is None:
        print("Processing failed.")
        return
        
    print(f"Predictions (Top 5):\n{df_preds[['馬番', 'probability', 'odds', 'expected_value']].head(5)}")
    
    print("\n--- Allocations ---")
    strategies = ['balance', 'formation']
    budgets = [1000, 5000, 10000]
    
    for strat in strategies:
        for bud in budgets:
            print(f"\n[Strategy: {strat}, Budget: {bud}]")
            recs = BettingAllocator.allocate_budget(df_preds, bud, strategy=strat)
            if not recs:
                print("No bets.")
            else:
                for r in recs:
                    print(f" - {r['bet_type']} ({r.get('method')}): {r.get('horse_numbers')} Amt:{r.get('total_amount') or r.get('amount')}")

if __name__ == "__main__":
    run_one_race()
