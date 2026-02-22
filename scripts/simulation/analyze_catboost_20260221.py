import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.simulation.verify_integrated_report import extract_bets_from_integrated_report

def main():
    report_path = "reports/prediction_20260221_integrated_v2.md"
    payout_path = "data/raw/payouts_20260221.pkl"
    
    if not os.path.exists(report_path) or not os.path.exists(payout_path):
        print("Required files not found.")
        return

    bets_by_race = extract_bets_from_integrated_report(report_path)
    
    with open(payout_path, 'rb') as f:
        payouts = pickle.load(f)
        
    print("=== CatBoost_BOX_sanrenpuku 分析 (2026/02/21) ===")
    
    total_races = 0
    total_bets = 0
    
    for rid, bets in sorted(bets_by_race.items()):
        cat_bets = [b for b in bets if 'CatBoost' in b['strategy']]
        if not cat_bets:
            continue
            
        total_races += 1
        total_bets += len(cat_bets)
        print(f"\n--- レースID: {rid} ---")
        if rid not in payouts:
            print("実績データなし")
            continue
            
        p = payouts[rid]
        
        actual_sanrenpuku = p.get('sanrenpuku', {})
        actual_tan = p.get('tan', {})
        
        print("【実績】")
        for k, v in actual_tan.items():
            print(f"  単勝: 馬番 {k} (配当: {v}円)")
        for k, v in list(actual_sanrenpuku.items())[:3]: # 表示を簡略化
            print(f"  3連複: {k} (配当: {v}円)")
            
        horses_set = set()
        for b in cat_bets:
            for num in b['combination']:
                horses_set.add(num)
                
        print(f"【CatBoost】買っていた馬番 (BOX): {sorted(list(horses_set))}")
        
        for act_comb in actual_sanrenpuku.keys():
            act_horses = set(act_comb)
            matches = act_horses.intersection(horses_set)
            missed = act_horses - horses_set
            if len(matches) == 3:
                print(f"  !! 的中 !! {act_comb} (配当: {actual_sanrenpuku[act_comb]}円)")
            else:
                print(f"  -> 不的中: 正解 {sorted(list(act_horses))} のうち {len(matches)}頭カバー (抜け: {sorted(list(missed))})")

    print(f"\n全体集計: {total_races}レース対象, {total_bets}点の買い目")

if __name__ == '__main__':
    main()
