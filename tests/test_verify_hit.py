
import pandas as pd
import sys
import os

# Add scripts directory to path to import verify_hit if it was in a module, 
# but verify_hit is inside simulate_2021_2025.py which is not a module.
# Check if I can import it or if I should just copy the logic.
# I will copy the function here to test it in isolation first.

def verify_hit(rec, returns_dict, race_id):
    # Logic copied from simulate_2021_2025.py
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None: return 0
    payout = 0
    rec_type = rec.get('bet_type') or rec.get('type')
    bet_horse_nums = set(rec.get('horse_numbers') or rec.get('horses', []))
    if not rec_type or not bet_horse_nums: return 0
    
    try:
        hits = race_rets[race_rets[0] == rec_type]
        for _, h in hits.iterrows():
            try:
                money = int(str(h[2]).replace(',','').replace('円',''))
                win_str = str(h[1]).replace('→','-')
                if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
                else: win_nums = [int(win_str)]
                is_hit = False
                
                method = rec.get('method')
                formation = rec.get('formation')
                
                if rec_type in ['単勝', '複勝']:
                    if win_nums[0] in bet_horse_nums: is_hit = True
                elif rec_type in ['馬連', 'ワイド', '3連複', '3連単']:
                    if method in ['流し', 'Formation', 'FORMATION'] and formation:
                        if rec_type == '3連複':
                            if len(formation) == 2:
                                axis = set(formation[0])
                                opponents = set(formation[1])
                                win_set = set(win_nums)
                                if axis.issubset(win_set):
                                    if (win_set - axis).issubset(opponents): is_hit = True
                        elif rec_type == '3連単':
                            if len(formation) == 3:
                                g1, g2, g3 = [set(x) for x in formation]
                                if win_nums[0] in g1 and win_nums[1] in g2 and win_nums[2] in g3:
                                    is_hit = True
                        elif rec_type == '馬連' or rec_type == 'ワイド':
                            if len(formation) == 2:
                                head = set(formation[0])
                                opps = set(formation[1])
                                print(f"DEBUG: {rec_type} Nagashi Check. Head={head} Opps={opps} Win={win_nums}")
                                
                                if not head.isdisjoint(set(win_nums)):
                                     other = set(win_nums) - head
                                     if other.issubset(opps): 
                                         is_hit = True
                                         print(f"DEBUG: HIT! Head={head} Other={other}")

                    elif method == 'BOX' or not method:
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                        
                if is_hit:
                    unit = rec.get('unit_amount', 100)
                    if unit == 0:
                        count = rec.get('points', rec.get('count', 1))
                        total_amt = rec.get('total_amount', rec.get('amount', 0))
                        unit = total_amt // count if count > 0 else 100
                    payout += money * (unit / 100)
            except Exception as e:
                print(f"Error in hit processing: {e}")
                pass
    except Exception as e:
        print(f"Error in race processing: {e}")
        pass
    return payout

# Test Data
def test_wide_nagashi():
    print("Testing Wide Nagashi...")
    # Mock returns: Wide 1-2 (200yen), 1-3 (300yen), 2-3 (400yen)
    # DataFrame structure: [bet_type, combination, payout]
    returns_df = pd.DataFrame([
        ['ワイド', '1-2', '200円'],
        ['ワイド', '1-3', '300円'],
        ['ワイド', '2-3', '400円']
    ])
    returns_dict = {'R1': returns_df}
    
    # Strat 1: Axis 1, Opponents [2, 3, 4, 5]
    # Should hit 1-2 and 1-3. Total payout: 200 + 300 = 500
    rec1 = {
        'type': 'ワイド',
        'method': '流し',
        'horses': [1, 2, 3, 4, 5],
        'formation': [[1], [2, 3, 4, 5]],
        'unit_amount': 100
    }
    
    payout1 = verify_hit(rec1, returns_dict, 'R1')
    print(f"Payout 1 (Exp 500): {payout1}")
    assert payout1 == 500

    # Strat 2: Axis 4, Opponents [1, 2, 3]
    # Should hit nothing (4 is not in 1-2, 1-3, 2-3)
    rec2 = {
        'type': 'ワイド',
        'method': '流し',
        'horses': [4, 1, 2, 3],
        'formation': [[4], [1, 2, 3]],
        'unit_amount': 100
    }
    
    payout2 = verify_hit(rec2, returns_dict, 'R1')
    print(f"Payout 2 (Exp 0): {payout2}")
    assert payout2 == 0

    # Strat 3: Axis 1, Opponents [4, 5]
    # Should hit nothing (1 is in 1-2, 1-3, but 2,3 are not in opps)
    rec3 = {
        'type': 'ワイド',
        'method': '流し',
        'horses': [1, 4, 5],
        'formation': [[1], [4, 5]],
        'unit_amount': 100
    }
    payout3 = verify_hit(rec3, returns_dict, 'R1')
    print(f"Payout 3 (Exp 0): {payout3}")
    assert payout3 == 0

    print("Wide Test Passed")

def test_umaren_nagashi():
    print("Testing Umaren Nagashi...")
    # Mock returns: Umaren 1-2 (1000yen)
    returns_df = pd.DataFrame([
        ['馬連', '1-2', '1,000円']
    ])
    returns_dict = {'R1': returns_df}
    
    # Strat 1: Axis 1, Opponents [2, 3]
    # Should hit 1-2. Payout: 1000
    rec1 = {
        'type': '馬連',
        'method': '流し',
        'horses': [1, 2, 3],
        'formation': [[1], [2, 3]],
        'unit_amount': 100
    }
    
    payout1 = verify_hit(rec1, returns_dict, 'R1')
    print(f"Payout 1 (Exp 1000): {payout1}")
    assert payout1 == 1000
    
    print("Umaren Test Passed")

if __name__ == "__main__":
    test_wide_nagashi()
    test_umaren_nagashi()
