"""
2025年シミュレーションスクリプト (LTR vs LGBM)
- 目的: 生成された予測データと収集した配当データを用いて、2025年全期間の収支シミュレーションを行う。
- 戦略:
    1. LTR: 単勝、馬連（Box/流し）
    2. LGBM: Box4（3連複4頭Box）など
    3. 予算: 各レース5000円
- データ:
    - 予測: data/processed/prediction_2025_{model}.csv
    - 配当: data/raw/payouts_2025.pkl
- 出力: reports/simulation_2025_comparison.md
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR

# ============= 設定 =============
PRED_LTR_PATH = "data/processed/prediction_2025_ltr.csv"
PRED_LGBM_PATH = "data/processed/prediction_2025_lgbm.csv"
PAYOUTS_PATH = os.path.join(RAW_DATA_DIR, "payouts_2025.pkl")
REPORT_PATH = "reports/simulation_2025_comparison.md"
START_DATE = '2025-01-01'
END_DATE = '2025-12-31'
BUDGET_PER_RACE = 5000
# ===============================

class SimulationRunner:
    def __init__(self):
        self.payouts = self._load_payouts()
        
    def _load_payouts(self):
        if not os.path.exists(PAYOUTS_PATH):
            print(f"Warning: {PAYOUTS_PATH} not found. Simulation will be limited to Win (if available in pred data).")
            return {}
        with open(PAYOUTS_PATH, 'rb') as f:
            return pickle.load(f)
            
    def get_payout(self, race_id, bet_type, horse_indices):
        """
        配当を取得する
        race_id: str
        bet_type: 'tan', 'umaren', 'wide', 'sanrenpuku', etc.
        horse_indices: list or tuple of horse numbers (1-based)
        """
        race_id = str(race_id)
        if race_id not in self.payouts:
            return 0
            
        race_payouts = self.payouts[race_id]
        if bet_type not in race_payouts:
            return 0
            
        # Key作成
        if bet_type in ['tan', 'fuku']:
            key = horse_indices[0]
        elif bet_type in ['umaren', 'wide', 'wakuren', 'sanrenpuku']:
            key = tuple(sorted(horse_indices))
        else: # umatan, sanrentan (order sensitive)
            key = tuple(horse_indices)
            
        return race_payouts.get(key, 0)

    def calculate_return(self, race_id: str, strategy_name: str, recommended_bets: list) -> tuple:
        """払戻計算 (Return, Hits)"""
        if race_id not in self.payouts:
            return 0, 0
            
        race_payouts = self.payouts[race_id]
        total_payout = 0
        hits = 0
        
        # Debug trace (1回だけ)
        debug = False
        if not hasattr(self, '_debug_done'):
             self._debug_done = False
        if not self._debug_done and recommended_bets:
             debug = True
             self._debug_done = True
             print(f"--- Debug Calculation for {race_id} ({strategy_name}) ---")
             print(f"Payout Keys (tan): {list(race_payouts.get('tan', {}).keys())}")

        for bet in recommended_bets:
            b_type = bet.get('type')
            amount = bet.get('amount', 0)
            
            # Key "horses" or "uma_ban"
            targets = bet.get('horses') or bet.get('uma_ban') or []
            if not targets: continue
            
            # --- Box4 Special Handling ---
            if b_type == 'box4_sanrenpuku':
                # 4C3 = 4 combinations
                combos = list(combinations(targets, 3))
                unit_amount = amount / len(combos) # Split amount? Or amount is total?
                # In strat_lgbm_box4, amount is 4800 (1200 per point).
                # So unit_amount calculation depends on definition. 
                # Assuming 'amount' in strat definition is TOTAL amount.
                
                for cb in combos:
                    # Check sanrenpuku payout
                    if 'sanrenpuku' not in race_payouts: continue
                    
                    key = tuple(sorted([int(x) for x in cb]))
                    p_dict = race_payouts['sanrenpuku']
                    
                    if key in p_dict:
                        pay = p_dict[key]
                        # Return = Payout * (UnitBet / 100)
                        total_payout += pay * (unit_amount / 100)
                        hits += 1
                        if debug: print(f"  Box4 Hit! Key:{key} Pay:{pay}")

            # --- Standard Handling ---
            elif b_type in race_payouts:
                p_dict = race_payouts[b_type]
                
                key = None
                if b_type in ['tan', 'fuku']:
                    if len(targets) == 1:
                        key = int(targets[0])
                else:
                    if b_type in ['umaren', 'wide', 'wakuren', 'sanrenpuku', 'sanrentan']:
                        key = tuple(sorted([int(x) for x in targets]))
                    else:
                        key = tuple([int(x) for x in targets])
                
                if debug:
                    print(f"Checking {b_type} Key: {key} (Raw:{targets})")
                
                if key in p_dict:
                    pay = p_dict[key]
                    total_payout += pay * (amount / 100)
                    hits += 1
                    if debug: print(f"  Hit! Pay:{pay}")
            
        return int(total_payout), hits

    def run(self):
        # Load Predictions
        print("Loading predictions...")
        df_ltr = pd.read_csv(PRED_LTR_PATH)
        df_lgbm = pd.read_csv(PRED_LGBM_PATH)
        
        print(f"Loaded Payouts: {len(self.payouts)} races.")
        if self.payouts:
            print(f"Sample Payout Keys: {list(self.payouts.keys())[:5]}")
            
        df_ltr['date'] = pd.to_datetime(df_ltr['date'])
        if 'race_id' in df_ltr.columns:
            df_ltr['race_id'] = df_ltr['race_id'].astype(str)
        if 'horse_number' in df_ltr.columns:
            df_ltr['horse_number'] = df_ltr['horse_number'].astype(int)
            
        df_lgbm['date'] = pd.to_datetime(df_lgbm['date'])
        if 'race_id' in df_lgbm.columns:
            df_lgbm['race_id'] = df_lgbm['race_id'].astype(str)
        if 'horse_number' in df_lgbm.columns:
            df_lgbm['horse_number'] = df_lgbm['horse_number'].astype(int)

        print(f"Sample PRED_LTR race_ids: {df_ltr['race_id'].head(3).tolist()}")
        print(f"Sample PRED_LGBM race_ids: {df_lgbm['race_id'].head(3).tolist()}")

        # ID Matching Check
        pred_ids = set(df_ltr['race_id'].unique())
        payout_ids = set(self.payouts.keys())
        common_ids = pred_ids & payout_ids
        print(f"Pred IDs: {len(pred_ids)}, Payout IDs: {len(payout_ids)}")
        print(f"Common IDs: {len(common_ids)}")
        if len(common_ids) == 0:
            print(f"No match! Pred Sample: {list(pred_ids)[:3]}")
            print(f"Payout Sample: {list(payout_ids)[:3]}")

        # Results Container
        results = {}
        
        # Strategies Definition
        # function(row_group) -> list of bets
        
        def strat_ltr_win1(df_race):
            # 単勝1点
            top = df_race.iloc[0]
            return [{'type': 'tan', 'horses': [top['horse_number']], 'amount': BUDGET_PER_RACE}]

        def strat_lgbm_win1(df_race):
            top = df_race.iloc[0]
            return [{'type': 'tan', 'horses': [top['horse_number']], 'amount': BUDGET_PER_RACE}]

        def strat_lgbm_box4(df_race):
            # 3連複4頭Box (上位4頭)
            # 予測スコア順
            if len(df_race) < 4: return []
            top4 = df_race['horse_number'].iloc[:4].tolist()
            # 4C3 = 4点。 5000円予算 -> 1点1200円 (4800円)
            return [{'type': 'box4_sanrenpuku', 'horses': top4, 'amount': 4800}]

        def strat_lgbm_box5(df_race):
            # 3連複5頭Box
            if len(df_race) < 5: return []
            top5 = df_race['horse_number'].iloc[:5].tolist()
            # 5C3 = 10点。 5000円 -> 1点500円
            combos = list(combinations(top5, 3))
            bets = []
            for cb in combos:
                bets.append({'type': 'sanrenpuku', 'horses': cb, 'amount': 500})
            return bets

        strategies = [
            {'name': 'LTR_01_Win(Top1)', 'df': df_ltr, 'func': strat_ltr_win1, 'score': 'ltr_score'},
            {'name': 'LGBM_01_Win(Top1)', 'df': df_lgbm, 'func': strat_lgbm_win1, 'score': 'prob_score'},
            {'name': 'LGBM_02_Box4(3renpuku)', 'df': df_lgbm, 'func': strat_lgbm_box4, 'score': 'prob_score'},
            {'name': 'LGBM_03_Box5(3renpuku)', 'df': df_lgbm, 'func': strat_lgbm_box5, 'score': 'prob_score'},
        ]
        
        for strat in strategies:
            entry = {'Races': 0, 'Bet': 0, 'Return': 0, 'Hit': 0}
            print(f"Simulating {strat['name']}...")
            
            df = strat['df']
            score_col = strat['score']
            func = strat['func']
            
            # Group by race
            groups = df.groupby('race_id')
            
            for race_id, group in tqdm(groups):
                # Sort by score
                group = group.sort_values(score_col, ascending=False)
                
                # Retrieve Bets
                bets = func(group)
                if not bets: continue
                
                # Calculate Cost & Return
                bet_amount = sum(b['amount'] for b in bets)
                ret_amount, hits = self.calculate_return(race_id, strat['name'], bets)
                
                entry['Races'] += 1
                entry['Bet'] += bet_amount
                entry['Return'] += ret_amount
                if hits > 0: entry['Hit'] += 1 # レース単位の的中 (Box4で複数Hitしても1とするか...今回は1でよい)
            
            # Summary stats
            entry['Net'] = entry['Return'] - entry['Bet']
            entry['ROI'] = (entry['Return'] / entry['Bet'] * 100) if entry['Bet'] > 0 else 0
            entry['HitRate'] = (entry['Hit'] / entry['Races'] * 100) if entry['Races'] > 0 else 0
            
            results[strat['name']] = entry
            
        # Report Output
        self._write_report(results)

    def _write_report(self, results):
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"# 2025年シミュレーションレポート (LTR vs LGBM)\n\n")
            f.write(f"データ日付: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"予算: {BUDGET_PER_RACE}円/レース (概算)\n\n")
            f.write("| 戦略 | レース数 | 購入額 | 払戻額 | 純利益 | 回収率 | 的中率 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            
            for name, res in results.items():
                f.write(f"| {name} | {res['Races']:,} | {res['Bet']:,} | {int(res['Return']):,} | {int(res['Net']):,} | {res['ROI']:.1f}% | {res['HitRate']:.1f}% |\n")
                
        print(f"Report saved to {REPORT_PATH}")
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            print(f.read())

if __name__ == "__main__":
    from datetime import datetime
    runner = SimulationRunner()
    runner.run()
