
import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# ============= 設定 =============
TARGET_DATE = '20260215'
PREDICTION_FILE = f"data/processed/prediction_{TARGET_DATE}.csv"
PAYOUT_FILE = f"data/raw/payouts_{TARGET_DATE}.pkl"
REPORT_FILE = f"reports/simulation_{TARGET_DATE}.md"
BUDGET_PER_RACE = 5000
# ===============================

class BettingSimulator:
    def __init__(self, predictions, payouts):
        self.predictions = predictions
        self.payouts = payouts
        self.results = []
        
    def run(self):
        race_ids = self.predictions['race_id'].unique()
        print(f"Simulating {len(race_ids)} races...")
        
        strategies = [
            {'name': 'LGBM_Box4', 'func': self.strategy_box4, 'model': 'lgbm', 'type': 'sanrenpuku'},
            {'name': 'LGBM_Box5', 'func': self.strategy_box5, 'model': 'lgbm', 'type': 'sanrenpuku'},
            {'name': 'LTR_Win',   'func': self.strategy_win,  'model': 'ltr',  'type': 'tan'},
        ]
        
        summary_data = []
        
        for strat in strategies:
            total_bet = 0
            total_return = 0
            hit_count = 0
            race_count = 0
            
            for rid in race_ids:
                race_pred = self.predictions[self.predictions['race_id'] == rid]
                payout = self.payouts.get(str(rid), {})
                
                # スコア順にソート
                score_col = 'prob_score' if strat['model'] == 'lgbm' else 'ltr_score'
                if score_col not in race_pred.columns: continue
                
                sorted_pred = race_pred.sort_values(score_col, ascending=False)
                
                # 買い目決定
                bet_list = strat['func'](sorted_pred)
                if not bet_list: continue
                
                # 投資額計算 (均等配分)
                cost_per_bet = BUDGET_PER_RACE // len(bet_list)
                if cost_per_bet < 100: cost_per_bet = 100 # 最低100円
                
                bet_amount = cost_per_bet * len(bet_list)
                return_amount = 0
                
                # 判定
                hit = False
                type_key = strat['type']
                
                if type_key in payout:
                    winning_combinations = payout[type_key] # {comb: payout}
                    
                    for my_bet in bet_list:
                        # my_betはtuple (馬番, ...)
                        # winning_combinationsのキーと比較
                        # 単勝の場合は (馬番,) か int かもしれないので注意
                        
                        # 3連複などの組み合わせ
                        if isinstance(my_bet, tuple) and len(my_bet) > 1:
                            s_bet = tuple(sorted(my_bet))
                            if s_bet in winning_combinations:
                                return_amount += (winning_combinations[s_bet] / 100) * cost_per_bet
                                hit = True
                        
                        # 単勝・複勝
                        else:
                            # 単勝キーはintの場合が多い
                            bet_num = my_bet[0] if isinstance(my_bet, tuple) else my_bet
                            if bet_num in winning_combinations:
                                return_amount += (winning_combinations[bet_num] / 100) * cost_per_bet
                                hit = True
                                
                total_bet += bet_amount
                total_return += return_amount
                if hit: hit_count += 1
                race_count += 1
                
            net = total_return - total_bet
            roi = (total_return / total_bet * 100) if total_bet > 0 else 0
            
            summary_data.append({
                'Strategy': strat['name'],
                'Races': race_count,
                'Bet': total_bet,
                'Return': int(total_return),
                'Net': int(net),
                'ROI': round(roi, 1),
                'HitRate': round(hit_count / race_count * 100, 1) if race_count > 0 else 0
            })
            
        return pd.DataFrame(summary_data)

    def strategy_win(self, df):
        # Top1 単勝
        top1 = df.iloc[0]['horse_number']
        return [(top1,)]

    def strategy_box4(self, df):
        # Top4 3連複Box (4C3 = 4点)
        top4 = df.head(4)['horse_number'].tolist()
        if len(top4) < 4: return []
        import itertools
        return list(itertools.combinations(top4, 3))

    def strategy_box5(self, df):
        # Top5 3連複Box (5C3 = 10点)
        top5 = df.head(5)['horse_number'].tolist()
        if len(top5) < 5: return []
        import itertools
        return list(itertools.combinations(top5, 3))


def main():
    print(f"Loading data for {TARGET_DATE}...")
    
    if not os.path.exists(PREDICTION_FILE):
        print(f"Prediction file not found: {PREDICTION_FILE}")
        return
        
    if not os.path.exists(PAYOUT_FILE):
        print(f"Payout file not found: {PAYOUT_FILE}")
        # 配当ファイルがない場合は実行中断 (スクレイピング待ち)
        return

    df_pred = pd.read_csv(PREDICTION_FILE)
    with open(PAYOUT_FILE, 'rb') as f:
        payouts = pickle.load(f)
        
    print(f"Loaded predictions: {len(df_pred)} rows")
    print(f"Loaded payouts: {len(payouts)} races")
    
    # Run Simulation
    sim = BettingSimulator(df_pred, payouts)
    df_summary = sim.run()
    
    print("\n=== Simulation Results ===")
    print(df_summary.to_markdown(index=False))
    
    # Save Report
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Simulation Report {TARGET_DATE}\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n## Details\n")
        f.write(f"- Budget: {BUDGET_PER_RACE} JPY/Race\n")
        f.write(f"- Races: {len(payouts)}\n")

if __name__ == "__main__":
    main()
