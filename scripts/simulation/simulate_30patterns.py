"""
30 Pattern Simulation Script
Budgets: [500, 1000, 2000, 3000, 5000]
Bet Types: ['単勝', '複勝', '馬連', 'ワイド', '3連複', '3連単']
Period: 2022-2025
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

def run_simulation_30patterns():
    # 1. Load Prediction Data
    sim_path = 'rolling_prediction_details_v2.csv'
    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found.")
        return
    
    print(f"Loading predictions from {sim_path}...")
    sim_df = pd.read_csv(sim_path)
    
    # Identify races by original_race_id (Convert to string for matching with return_df index)
    if 'original_race_id' not in sim_df.columns:
        print("Error: original_race_id column missing in prediction data.")
        return
    sim_df['original_race_id'] = sim_df['original_race_id'].astype(str)

    # 2. Load Payout Data
    return_path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
    if not os.path.exists(return_path):
        print(f"Error: {return_path} not found.")
        return
    
    print(f"Loading payouts from {return_path}...")
    return_df = pd.read_pickle(return_path)
    # Ensure return_df index is string level 0
    if isinstance(return_df.index, pd.MultiIndex):
        # Already MultiIndex, ensure level 0 is string
        pass # Assuming it's already string from native netkeiba scraping
    
    # 3. Define Patterns
    budgets = [500, 1000, 2000, 3000, 5000]
    bet_types = ['単勝', '複勝', '馬連', 'ワイド', '3連複', '3連単']
    
    patterns = []
    for b in budgets:
        for t in bet_types:
            patterns.append({'budget': b, 'type': t})
            
    # Results dictionary
    # Key: (budget, type) -> {'invest': 0, 'return': 0, 'hits': 0, 'race_count': 0}
    stats = {}
    for p in patterns:
        stats[(p['budget'], p['type'])] = {'invest': 0, 'return': 0, 'hits': 0, 'race_count': 0}

    # Group by race_id
    grouped = sim_df.groupby('original_race_id')
    
    print(f"Simulating {len(patterns)} patterns over {len(grouped)} races...")
    
    # Performance optimization: pre-filter return_df? 
    # Usually return_df lookup is fast enough if indexed.
    
    for rid, race_df in tqdm(grouped):
        # 1. Check if payout exists for this race
        if rid not in return_df.index:
            continue
            
        payouts = return_df.loc[rid]
        
        # 2. Preparation for Allocation
        # BettingAllocator expects 'horse_number' and 'score' (probability)
        # Rename to match expected format
        alloc_input = race_df.rename(columns={'馬番': 'horse_number', 'score': 'probability'})
        
        # 3. For each pattern, allocate and test
        for p in patterns:
            budget = p['budget']
            b_type = p['type']
            
            # Generate recommended bets for this pattern
            bets = BettingAllocator.allocate_budget(alloc_input, budget, allowed_types=[b_type])
            
            if not bets:
                continue
                
            # Calculate return for these bets
            # strategy_composite calculations use unit_amount (handled inside)
            # but usually they return concrete amounts.
            
            # Convert BettingAllocator output format back to what strategy_composite expects if necessary
            # Actually CompositeBettingStrategy.calculate_return takes a list of bet dicts
            # BettingAllocator code uses format:
            # { 'bet_type', 'method', 'combination', 'description', 'points', 'unit_amount', 'total_amount', 'horse_numbers' }
            # Let's ensure format is compatible.
            
            patt_return = 0
            try:
                # We can't use CompositeBettingStrategy.calculate_return directly because it expects 'type' and 'umaban'?
                # Let's re-verify strategy_composite.py requirement.
                # In previous simulate_multibet.py, bets had 'type' and 'horse_no' or 'combination'.
                
                # Manual return calculation logic to be safe and fast:
                for bet in bets:
                    invest = bet['total_amount']
                    stats[(budget, b_type)]['invest'] += invest
                    
                    # Calculate payout
                    amt = BettingAllocator._manual_calc_payout(bet, payouts)
                    if amt > 0:
                        patt_return += amt
                
                stats[(budget, b_type)]['return'] += patt_return
                if patt_return > 0:
                    stats[(budget, b_type)]['hits'] += 1
                    
                stats[(budget, b_type)]['race_count'] += 1
                
            except Exception as e:
                # print(f"Error in race {rid}, pattern {p}: {e}")
                pass

    # 4. Summarize Results
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
    output_path = 'simulation_30patterns_results.csv'
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n=== Simulation Summary (30 Patterns) ===")
    # Format and print clearly
    summary_pivot = df_results.pivot(index='Budget', columns='Type', values='RecoveryRate')
    print("\nRecovery Rate (%) by Budget and Type:")
    print(summary_pivot)
    
    print(f"\nFinal results saved to {output_path}")

# Helper for manual payout calculation
def _manual_calc_payout(bet, payouts):
    b_type = bet['bet_type']
    method = bet['method']
    unit = bet['unit_amount']
    horses = bet['horse_numbers']
    
    import itertools
    from modules.strategy_composite import CompositeBettingStrategy
    
    expanded_bets = []
    
    if method == 'SINGLE':
        # Horse number is in combination string
        expanded_bets.append({
            'type': b_type,
            'combo': str(bet['combination']),
            'amount': unit
        })
    elif method == 'BOX':
        # Generate all possible combinations
        # Use sorted horses for non-ordered types
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

# Monkey patch helper into BettingAllocator for the script's use
BettingAllocator._manual_calc_payout = staticmethod(_manual_calc_payout)

if __name__ == '__main__':
    run_simulation_30patterns()
