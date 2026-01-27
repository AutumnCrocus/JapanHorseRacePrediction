import pandas as pd
import sys
import os

# Add module path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.betting_allocator import BettingAllocator
from modules.strategy_composite import CompositeBettingStrategy
from modules.constants import RAW_DATA_DIR

def debug_payout():
    # 1. Load Data
    sim_path = 'rolling_prediction_details_v2.csv'
    return_path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
    
    sim_df = pd.read_csv(sim_path)
    sim_df['original_race_id'] = sim_df['original_race_id'].astype(str)
    return_df = pd.read_pickle(return_path)
    
    # 2. Pick a Graded Race that we know the Top 1 Horse WON
    # We can use the 'rank' column in sim_df
    
    # Sort by score descending within each race
    sim_df = sim_df.sort_values(['original_race_id', 'score'], ascending=[True, False])
    
    # Filter for Rank 1 (Top Score horse won)
    winners = sim_df[(sim_df.groupby('original_race_id')['score'].transform('max') == sim_df['score']) & (sim_df['rank'] == 1)]
    
    if winners.empty:
        print("No races found where Top 1 Score horse won.")
        return
        
    target_race_id = winners.iloc[0]['original_race_id']
    target_horse_no = winners.iloc[0]['馬番']
    
    print(f" debugging race: {target_race_id}")
    print(f" Winner (Top Score): {target_horse_no}")
    
    # 3. Simulate Logic
    race_df = sim_df[sim_df['original_race_id'] == target_race_id].copy()
    alloc_input = race_df.rename(columns={'馬番': 'horse_number', 'score': 'probability'})
    
    # Generate Single Bet
    # Budget 500, Type '単勝'
    bets = BettingAllocator.allocate_budget(alloc_input, 500, allowed_types=['単勝'])
    
    print("\nGenerated Bets:")
    for b in bets:
        print(b)
        
    # 4. Calculate Payout using Manual Logic
    payouts = return_df.loc[target_race_id]
    print("\nPayouts Data:")
    print(payouts)
    
    # Manual Calc logic from simulate script
    def manual_calc(bet, payouts):
        b_type = bet['bet_type']
        method = bet['method']
        unit = bet['unit_amount']
        horses = bet['horse_numbers']
        
        expanded_bets = []
        if method == 'SINGLE':
            combo = str(bet['combination']) # This is where 'combination' comes from format_recommendations
            # format_recommendations does: combo_str = "-".join(map(str, r['horses']))
            
            expanded_bets.append({
                'type': b_type,
                'combo': combo,
                'amount': unit
            })
            
        print(f"\nExpanded Bets for Calc: {expanded_bets}")
        amt = CompositeBettingStrategy.calculate_return(expanded_bets, payouts)
        return amt

    total_return = 0
    for b in bets:
        ret = manual_calc(b, payouts)
        total_return += ret
        
    print(f"\nTotal Return Calculated: {total_return}")
    
    if total_return > 0:
        print("SUCCESS: Payout calculated correctly.")
    else:
        print("FAILURE: Payout is 0 despite winner picked.")
        
        # Debug why
        # Check matching logic in strategy_composite?
        # Let's inspect payouts['単勝'] structure
        win_pay = payouts.get('単勝')
        print(f"\nWin Payout Structure: {win_pay}")

if __name__ == "__main__":
    debug_payout()
