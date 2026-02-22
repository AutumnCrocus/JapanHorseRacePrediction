import os
import sys
import pickle
import pandas as pd
import re
import itertools
from tabulate import tabulate

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# ============= 設定 =============
TARGET_DATE = '20260221'
REPORT_FILE = f"reports/prediction_{TARGET_DATE}_integrated_v2.md"
PAYOUT_FILE = f"data/raw/payouts_{TARGET_DATE}.pkl"
# ===============================

def parse_horse_numbers(umaban_str):
    """
    "1 - 2" や "1" などの文字列をタプル (1, 2) や (1,) に変換する。
    " → " 区切りにも対応。
    """
    clean_str = str(umaban_str).replace(' - ', '-').replace(' → ', '-').replace(' ', '-')
    nums = re.findall(r'\d+', clean_str)
    if nums:
        return tuple(int(n) for n in nums)
    return None

def extract_bets_from_integrated_report(report_path):
    bets_by_race = {}
    current_race_id = None
    
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found")
        return bets_by_race

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by races
    races = re.split(r'\n## ', content)
    for race_block in races:
        if not race_block.strip():
            continue
            
        m_race = re.match(r'(.*?)\s*\((\d+)\)', race_block)
        if not m_race: continue
        race_id = m_race.group(2)
        bets_by_race[race_id] = []
        
        # --- 既存モデルのパース ---
        # "### 【既存モデルの推奨買い目】" から "### 【CatBoost" もしくは最後まで
        existing_sec = re.search(r'### 【既存モデルの推奨買い目】(.*?)(?:### 【CatBoost|\Z)', race_block, re.DOTALL)
        if existing_sec:
            existing_text = existing_sec.group(1)
            current_model_strategy = None
            for line in existing_text.split('\n'):
                line = line.strip()
                m_model = re.match(r'###\s+モデル:\s+(.*?)\s+/\s+戦略:\s+(.*?)\s+', line)
                if m_model:
                    model = m_model.group(1)
                    strategy = m_model.group(2)
                    current_model_strategy = f"{model}_{strategy}"
                    continue

                if line.startswith('- ') and current_model_strategy:
                    m_bet = re.match(r'-\s+(.*?)\s+(BOX|流し|SINGLE)\s+\((.*?)\):\s+(\d+)円\s+\((\d+)点\)', line)
                    if m_bet:
                        kind = m_bet.group(1)
                        method = m_bet.group(2)
                        horses_str = m_bet.group(3)
                        total_amount = int(m_bet.group(4))
                        num_bets = int(m_bet.group(5))
                        
                        if total_amount <= 0: continue
                        amount_per_bet = total_amount // num_bets

                        type_key = None
                        if '単勝' in kind: type_key = 'tan'
                        elif '複勝' in kind: type_key = 'fuku'
                        elif '枠連' in kind: type_key = 'wakuren'
                        elif '馬連' in kind: type_key = 'umaren'
                        elif 'ワイド' in kind: type_key = 'wide'
                        elif '馬単' in kind: type_key = 'umatan'
                        elif '3連複' in kind: type_key = 'sanrenpuku'
                        elif '3連単' in kind: type_key = 'sanrentan'
                        
                        if not type_key: continue

                        bet_combinations = []
                        if method == 'SINGLE':
                            nums = parse_horse_numbers(horses_str)
                            if nums: bet_combinations.append(nums)
                        elif method == 'BOX':
                            clean_str = horses_str.replace('BOX', '').strip()
                            nums = parse_horse_numbers(clean_str)
                            if nums:
                                if type_key == 'sanrenpuku':
                                    bet_combinations = list(itertools.combinations(nums, 3))
                                elif type_key in ['umaren', 'wide']:
                                    bet_combinations = list(itertools.combinations(nums, 2))
                        elif method == '流し':
                            m_axis = re.search(r'軸:(.*?)\s*-\s*相手:(.*)', horses_str)
                            if m_axis:
                                axis_nums = parse_horse_numbers(m_axis.group(1))
                                opp_nums = parse_horse_numbers(m_axis.group(2))
                                if axis_nums and opp_nums:
                                    if type_key in ['umaren', 'wide'] and len(axis_nums) == 1:
                                        axis = axis_nums[0]
                                        bet_combinations = [(axis, opp) for opp in opp_nums]
                                    elif type_key == 'sanrenpuku':
                                        if len(axis_nums) == 1:
                                            axis = axis_nums[0]
                                            bet_combinations = [(axis, p1, p2) for p1, p2 in itertools.combinations(opp_nums, 2)]
                                        elif len(axis_nums) == 2:
                                            a1, a2 = axis_nums
                                            bet_combinations = [(a1, a2, opp) for opp in opp_nums]

                        if type_key not in ['umatan', 'sanrentan']:
                            bet_combinations = [tuple(sorted(c)) for c in bet_combinations]

                        for comb in bet_combinations:
                            bets_by_race[race_id].append({
                                'strategy': current_model_strategy,
                                'type_key': type_key,
                                'combination': comb,
                                'amount': amount_per_bet
                            })

        # --- CatBoost モデルのパース ---
        catboost_sec = re.search(r'### 【CatBoost詳細分析.*?】(.*?)(?:\n---|\\n## |\Z)', race_block, re.DOTALL)
        if catboost_sec:
            cat_text = catboost_sec.group(1)
            if '平均EVが閾値(2.5)未満のため' not in cat_text:
                rec_sec = re.search(r'#### 買い目推奨:(.*?)(?:\n\n|\Z)', cat_text, re.DOTALL)
                if rec_sec:
                    for line in rec_sec.group(1).strip().split('\n'):
                        line = line.strip()
                        if not line.startswith('- **'): continue
                        
                        m = re.search(r'- \*\*([^\\*]+)\*\*: \[([\d, ]+)\] \((\d+)円\)', line)
                        if not m: continue
                        
                        method_raw = m.group(1)
                        horses_raw = m.group(2)
                        total_amount = int(m.group(3))
                        horses_list = [int(h.strip()) for h in horses_raw.split(',')]
                        
                        type_key = 'sanrenpuku'
                        strat_name = f"CatBoost_{method_raw}_sanrenpuku"
                        
                        bet_combinations = []
                        if method_raw == 'BOX':
                            bet_combinations = list(itertools.combinations(horses_list, 3))
                        
                        if bet_combinations:
                            amount_per_bet = total_amount // len(bet_combinations)
                            bet_combinations = [tuple(sorted(c)) for c in bet_combinations]
                            
                            for comb in bet_combinations:
                                bets_by_race[race_id].append({
                                    'strategy': strat_name,
                                    'type_key': type_key,
                                    'combination': comb,
                                    'amount': amount_per_bet
                                })

    return bets_by_race

def verify_bets(bets_by_race, payouts):
    strategy_metrics = {}

    for race_id, bets in bets_by_race.items():
        if race_id not in payouts:
            continue
            
        race_payout = payouts[race_id]
        
        for bet in bets:
            strat = bet['strategy']
            if strat not in strategy_metrics:
                strategy_metrics[strat] = {'bet': 0, 'return': 0, 'hit_count': 0, 'total_bets': 0}
            
            amt = bet['amount']
            strategy_metrics[strat]['bet'] += amt
            strategy_metrics[strat]['total_bets'] += 1
            
            type_key = bet['type_key']
            comb = bet['combination']
            
            if type_key in race_payout:
                winning_combs = race_payout[type_key]
                hit = False
                payout_amt = 0
                
                if len(comb) == 1 and type_key in ['tan', 'fuku']:
                    check_key = comb[0]
                    if check_key in winning_combs:
                        hit = True
                        payout_amt = winning_combs[check_key]
                else:
                    if comb in winning_combs:
                        hit = True
                        payout_amt = winning_combs[comb]
                        
                if hit:
                    ret = (payout_amt / 100) * amt
                    strategy_metrics[strat]['return'] += ret
                    strategy_metrics[strat]['hit_count'] += 1

    summary_data = []
    for strat, metrics in strategy_metrics.items():
        total_bet = metrics['bet']
        total_ret = metrics['return']
        net = total_ret - total_bet
        roi = (total_ret / total_bet * 100) if total_bet > 0 else 0
        hit_rate = (metrics['hit_count'] / metrics['total_bets'] * 100) if metrics['total_bets'] > 0 else 0
        
        summary_data.append({
            'Strategy': strat,
            '総投資(円)': total_bet,
            '回収額(円)': int(total_ret),
            '収支(円)': int(net),
            '回収率(%)': round(roi, 1),
            '的中回数': metrics['hit_count'],
            '総ベット数(点)': metrics['total_bets'],
            '的中率(%)': round(hit_rate, 2)
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('回収率(%)', ascending=False)
    
    return df_summary

def main():
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = TARGET_DATE

    report_path = f"reports/prediction_{target_date}_integrated_v2.md"
    payout_path = f"data/raw/payouts_{target_date}.pkl"

    print(f"Reading Report: {report_path}")
    bets_by_race = extract_bets_from_integrated_report(report_path)
    
    total_races = len([r for r in bets_by_race.values() if r])
    total_bets = sum(len(b) for b in bets_by_race.values())
    print(f"Extracted {total_bets} bets from {total_races} races.")
    
    print(f"Reading Payouts: {payout_path}")
    if not os.path.exists(payout_path):
        print("Error: Payout file not found.")
        return
        
    with open(payout_path, 'rb') as f:
        payouts = pickle.load(f)
    print(f"Payout data contains {len(payouts)} races.")

    df_summary = verify_bets(bets_by_race, payouts)
    
    print(f"\n=== {target_date} 統合レポート成績集計 ===")
    print(tabulate(df_summary, headers='keys', tablefmt='pipe', showindex=False))

    out_csv = f"reports/verify_integrated_result_{target_date}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_summary.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV to: {out_csv}")

if __name__ == "__main__":
    main()
