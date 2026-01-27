"""
Graded Race Simulation Script (30 Patterns)
Filters races by Prize Money >= 3000 (approx. Graded Class)
Run 30 betting patterns on these races.
Period: Based on available prediction data (likely 2022-2025)
"""
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.betting_allocator import BettingAllocator
from modules.strategy_composite import CompositeBettingStrategy
from modules.constants import RAW_DATA_DIR

def run_simulation_graded():
    # 1. Load Prediction Data
    sim_path = 'rolling_prediction_details_v2.csv'
    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found.")
        return
    
    print(f"Loading predictions from {sim_path}...")
    sim_df = pd.read_csv(sim_path)
    if 'original_race_id' not in sim_df.columns:
        print("Error: original_race_id column missing.")
        return
    sim_df['original_race_id'] = sim_df['original_race_id'].astype(str)

    # 2. Load Results for Prize Filtering
    results_path = os.path.join(RAW_DATA_DIR, 'results.pickle')
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    print(f"Loading results from {results_path} for prize info...")
    results_df = pd.read_pickle(results_path)
    
    # Clean prize data
    if '賞金 （万円）' in results_df.columns:
        if results_df['賞金 （万円）'].dtype == object:
            results_df['prize_clean'] = pd.to_numeric(results_df['賞金 （万円）'], errors='coerce').fillna(0)
        else:
            results_df['prize_clean'] = results_df['賞金 （万円）'].fillna(0)
    else:
        print("Error: '賞金 （万円）' column not found in results.")
        return

    # 3. Filter Graded Races (Prize >= 3000)
    print("Identifying Graded Races (Max Prize >= 3000)...")
    max_prizes = results_df.groupby(level=0)['prize_clean'].max()
    graded_race_ids = max_prizes[max_prizes >= 3000].index.astype(str)
    
    print(f"Total Graded Races Found: {len(graded_race_ids)}")
    
    # Filter sim_df
    sim_df = sim_df[sim_df['original_race_id'].isin(graded_race_ids)]
    print(f"Predictions reduced to {len(sim_df)} rows (Graded Races Only).")

    if len(sim_df) == 0:
        print("No races left after filtering.")
        return

    # 4. Load Payout Data
    return_path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
    if not os.path.exists(return_path):
        print(f"Error: {return_path} not found.")
        return
    
    print(f"Loading payouts from {return_path}...")
    return_df = pd.read_pickle(return_path)
    
    # 5. Define Patterns
    budgets = [500, 1000, 2000, 3000, 5000]
    bet_types = ['単勝', '複勝', '馬連', 'ワイド', '3連複', '3連単']
    
    patterns = []
    for b in budgets:
        for t in bet_types:
            patterns.append({'budget': b, 'type': t})
            
    stats = {}
    for p in patterns:
        stats[(p['budget'], p['type'])] = {'invest': 0, 'return': 0, 'hits': 0, 'race_count': 0}

    # Group by race_id
    grouped = sim_df.groupby('original_race_id')
    
    print(f"Simulating {len(patterns)} patterns over {len(grouped)} Graded races...")
    
    for rid, race_df in tqdm(grouped):
        if rid not in return_df.index:
            continue
            
        payouts = return_df.loc[rid]
        
        # Prepare Allocator Input
        alloc_input = race_df.rename(columns={'馬番': 'horse_number', 'score': 'probability'})
        
        for p in patterns:
            budget = p['budget']
            b_type = p['type']
            
            # Generate bets
            bets = BettingAllocator.allocate_budget(alloc_input, budget, allowed_types=[b_type])
            
            if not bets:
                continue
                
            patt_return = 0
            try:
                # Manual Calc
                for bet in bets:
                    invest = bet['total_amount']
                    stats[(budget, b_type)]['invest'] += invest
                    
                    amt = _manual_calc_payout(bet, payouts)
                    if amt > 0:
                        patt_return += amt
                
                stats[(budget, b_type)]['return'] += patt_return
                if patt_return > 0:
                    stats[(budget, b_type)]['hits'] += 1
                    
                stats[(budget, b_type)]['race_count'] += 1
                
            except Exception as e:
                pass

    # 6. Summarize Results
    result_list = []
    for (budget, b_type), d in stats.items():
        invest = d['invest']
        ret = d['return']
        recov = (ret / invest * 100) if invest > 0 else 0
        hit_rate = (d['hits'] / d['race_count'] * 100) if d['race_count'] > 0 else 0
        
        result_list.append({
            'Budget': budget,
            'Type': b_type,
            'Invested': invest,
            'Returned': ret,
            'RecoveryRate': recov,
            'HitRate': hit_rate,
            'Races': d['race_count']
        })
        
    df_results = pd.DataFrame(result_list)
    output_path = 'graded_race_simulation_results.csv'
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n=== Graded Race Simulation Summary ===")
    summary_pivot = df_results.pivot(index='Budget', columns='Type', values='RecoveryRate')
    print("\nRecovery Rate (%) by Budget and Type:")
    print(summary_pivot)
    
    # Detect best pattern
    best_row = df_results.loc[df_results['RecoveryRate'].idxmax()]
    print(f"\nBest Pattern: Budget={best_row['Budget']}, Type={best_row['Type']}")
    print(f"Recovery: {best_row['RecoveryRate']:.2f}%, Hit Rate: {best_row['HitRate']:.2f}%")
    
    print(f"\nFinal results saved to {output_path}")

# Helper for manual payout calculation (same as before)
def _manual_calc_payout(bet, payouts):
    b_type = bet['bet_type']
    method = bet['method']
    points = bet['points']
    if points > 0:
        unit = bet['total_amount'] // points
    else:
        unit = 0
    
    horses = bet['horse_numbers']
    
    import itertools
    from modules.strategy_composite import CompositeBettingStrategy
    
    expanded_bets = []
    
    if method == 'SINGLE':
        expanded_bets.append({
            'type': b_type,
            'combo': str(bet['combination']),
            'amount': unit
        })
    elif method == 'BOX':
        h_nums = sorted(horses)
        if b_type in ['馬連', 'ワイド', '枠連']:
            for c in itertools.combinations(h_nums, 2):
                expanded_bets.append({'type': b_type, 'combo': f"{c[0]}-{c[1]}", 'amount': unit})
        elif b_type == '馬単':
            for p in itertools.permutations(horses, 2):
                expanded_bets.append({'type': b_type, 'combo': f"{p[0]}→{p[1]}", 'amount': unit})
        elif b_type == '3連複':
            for c in itertools.combinations(h_nums, 3):
                expanded_bets.append({'type': b_type, 'combo': f"{c[0]}-{c[1]}-{c[2]}", 'amount': unit})
        elif b_type == '3連単':
            for p in itertools.permutations(horses, 3):
                expanded_bets.append({'type': b_type, 'combo': f"{p[0]}→{p[1]}→{p[2]}", 'amount': unit})
                
    if not expanded_bets:
        return 0
        
    return CompositeBettingStrategy.calculate_return(expanded_bets, payouts)

# Monkey patch
BettingAllocator._manual_calc_payout = staticmethod(_manual_calc_payout)

if __name__ == '__main__':
    run_simulation_graded()
