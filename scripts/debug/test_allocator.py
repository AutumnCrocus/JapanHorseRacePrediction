"""
Betting Allocator 検証スクリプト
異なる予算条件下での推奨買い目生成ロジックをテストする。
"""
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.betting_allocator import BettingAllocator

def test_allocation():
    # ダミー予測データ (上位馬)
    data = [
        {'horse_number': 1, 'probability': 0.60}, # High score, top pick
        {'horse_number': 2, 'probability': 0.30},
        {'horse_number': 3, 'probability': 0.15},
        {'horse_number': 4, 'probability': 0.10},
        {'horse_number': 5, 'probability': 0.08},
        {'horse_number': 6, 'probability': 0.05},
        {'horse_number': 7, 'probability': 0.02},
        {'horse_number': 8, 'probability': 0.01},
    ]
    df_preds = pd.DataFrame(data)
    
    budgets = [1000, 2000, 3000, 5000, 10000]
    
    print("=== Betting Allocator Verification ===")
    
    for b in budgets:
        print(f"\n[Budget: ¥{b}]")
        recs = BettingAllocator.allocate_budget(df_preds, b)
        
        total_invest = 0
        if not recs:
            print("  No recommendations generated.")
            continue
            
        for r in recs:
            print(f"  - {r['bet_type']:<5} {r['method']:<6} {r['combination']:<10} {r['description']:<10} ¥{r['total_amount']}")
            total_invest += r['total_amount']
            
        print(f"  Total Investment: ¥{total_invest}")
        if total_invest > b:
            print("  [ERROR] Over budget!")
        else:
            print("  [OK] Within budget.")

if __name__ == "__main__":
    test_allocation()
