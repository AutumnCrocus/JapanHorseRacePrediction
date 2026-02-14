import sys
import os
import re
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

# モジュールパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.scraping import Return
from modules.constants import HEADERS

# ----------------------------------------------------------------
# モンキーパッチ: 当日レース結果取得用 (evaluate_20260214.pyから移植)
# ----------------------------------------------------------------
def scrape_single_return_live(race_id, session):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        response.encoding = "EUC-JP"
        if response.status_code != 200:
            return race_id, None

        soup = BeautifulSoup(response.text, "lxml")
        payback_div = soup.find("div", {"class": "PayBackHost"}) or \
                      soup.find("div", {"class": "RaceResult_Return"}) or \
                      soup.find("div", {"class": "Full_PayBack_List"}) or \
                      soup.find("table", {"class": "PayBack_Table"})

        if payback_div:
            dfs = pd.read_html(StringIO(str(payback_div)))
            if dfs:
                merged_df = pd.concat(dfs)
                return race_id, merged_df
        
        dfs = pd.read_html(StringIO(response.text))
        for df in dfs:
            if "3連複" in str(df.values) or "３連複" in str(df.values):
                return race_id, df
        return race_id, None
    except Exception:
        return race_id, None

Return._scrape_single_return = scrape_single_return_live

# ----------------------------------------------------------------
# シミュレーションロジック
# ----------------------------------------------------------------

class Strategy:
    def __init__(self, name):
        self.name = name
    
    def decide_bets(self, df_race):
        """
        df_race: 特徴量や予測スコアが入ったDataFrame
        return: list of dict {'type': 'box', 'horses': [...], 'cost': ...}
        """
        raise NotImplementedError

class BoxStrategy(Strategy):
    def __init__(self, n_horses):
        super().__init__(f"{n_horses}頭Box")
        self.n_horses = n_horses
        
    def decide_bets(self, df_race):
        # スコア降順にソート済みと仮定、あるいはここでソート
        df = df_race.sort_values('ltr_score', ascending=False)
        top_n = df.head(self.n_horses)
        horses = top_n['horse_number'].tolist()
        
        # n頭BOXの点数計算: nC3
        combinations = 0
        if len(horses) >= 3:
            import math
            combinations = math.comb(len(horses), 3)
            
        cost = combinations * 100 # 1点100円
        
        return [{'type': 'box_sanrenpuku', 'horses': horses, 'cost': cost}]

class OddsFilterStrategy(Strategy):
    def __init__(self, n_horses, min_odds):
        super().__init__(f"{n_horses}頭Box (単勝{min_odds}倍未満除外)")
        self.n_horses = n_horses
        self.min_odds = min_odds
        
    def decide_bets(self, df_race):
        df = df_race.sort_values('ltr_score', ascending=False)
        
        # 上位候補を取得（余裕を持って多めに取る）
        candidates = df.head(self.n_horses + 5)
        
        selected_horses = []
        for _, row in candidates.iterrows():
            if len(selected_horses) >= self.n_horses:
                break
            
            # オッズフィルター: 人気しすぎている馬を除外？いいや、min_odds未満を除外
            # min_odds=2.0なら、単勝1.x倍の馬は買わない（穴狙い）
            if row['win_odds'] >= self.min_odds:
                selected_horses.append(row['horse_number'])
                
        # もし頭数が足りなければ、オッズ無視して上位から埋める？
        # ここでは厳格にフィルター適用して、足りなければ買わない、あるいは少頭数BOXにする
        
        if len(selected_horses) < 3:
            return [{'type': 'box_sanrenpuku', 'horses': [], 'cost': 0}]
            
        combinations = 0
        import math
        combinations = math.comb(len(selected_horses), 3)
        cost = combinations * 100
        
        return [{'type': 'box_sanrenpuku', 'horses': selected_horses, 'cost': cost}]

class FormationStrategy(Strategy):
    def __init__(self, axis_count, flow_count):
        super().__init__(f"Fm {axis_count}-{flow_count}")
        self.axis_count = axis_count
        self.flow_count = flow_count
        
    def decide_bets(self, df_race):
        df = df_race.sort_values('ltr_score', ascending=False)
        
        axis_horses = df.head(self.axis_count)['horse_number'].tolist()
        flow_horses = df.iloc[self.axis_count:self.axis_count+self.flow_count]['horse_number'].tolist()
        
        # フォーメーション: 軸 - 相手 - 相手
        # 3連複フォーメーション 1頭軸流し: Axis1 - (Axis+Flow) - (Axis+Flow)
        # ここではシンプルに「軸1頭 - 相手N頭」の流しについて計算
        # 点数: nC2 (相手から2頭選ぶ)
        
        if self.axis_count == 1:
            n_flow = len(flow_horses)
            if n_flow < 2:
                return [{'type': 'formation', 'horses': [], 'cost': 0}]
            import math
            combinations = math.comb(n_flow, 2)
            cost = combinations * 100
            
            # ベッティング情報の形式をどうするか
            # 評価関数calculate_payoutはBOX前提だった。
            # フォーメーション対応が必要だが、ここでは簡易的に「買い目リスト」を展開して持つか、
            # あるいはcalculate_payoutを改造するか。
            # 今はBoxStrategyと同じ形式で返すのは無理なので、特別扱いが必要。
            
            # 手抜き実装: Formationはまだ複雑なので、今回はBox戦略のバリエーションに集中する。
            return [{'type': 'formation', 'horses': [], 'cost': 0}]
        
        return [{'type': 'formation', 'horses': [], 'cost': 0}]


def calculate_payout_any(bet_info, result_df):
    """汎用払い戻し計算"""
    if result_df is None or result_df.empty:
        return 0, False
        
    total_return = 0
    hit = False
    
    try:
        if bet_info['type'] == 'box_sanrenpuku':
            bet_horses = set(bet_info['horses'])
            if len(bet_horses) < 3:
                return 0, False
                
            for _, row in result_df.iterrows():
                val0 = str(row.iloc[0])
                if '3連複' not in val0:
                    continue
                umaban_str = str(row.iloc[1])
                winning_horses = [int(x) for x in re.findall(r'\d+', umaban_str)]
                payout_str = str(row.iloc[2])
                payout = int(re.sub(r'[^\d]', '', payout_str))
                
                if set(winning_horses).issubset(bet_horses):
                    total_return += payout
                    hit = True
    except Exception:
        pass
        
    return total_return, hit


def main():  # noqa: C901
    # データ読み込み
    csv_path = "data/processed/prediction_20260214_ltr_full.csv"
    if not os.path.exists(csv_path):
        print("Error: Prediction data not found. Run predict script first.")
        return
        
    df_all = pd.read_csv(csv_path)
    race_ids = df_all['race_id'].unique()
    print(f"Loaded prediction data for {len(race_ids)} races.")
    
    # 払い戻しデータ取得
    print("Fetching race results...")
    # csvからレースIDリストを作る（文字列型にする）
    race_id_list = [str(rid) for rid in race_ids]
    results_db = Return.scrape(race_id_list)
    
    # 戦略定義
    strategies = [
        BoxStrategy(3), # 1点
        BoxStrategy(4), # 4点
        BoxStrategy(5), # 10点
        BoxStrategy(6), # 20点
        BoxStrategy(7), # 35点
        # 人気薄狙い: 単勝10倍以上の馬のみ、スコア上位から選んで5頭Box
        OddsFilterStrategy(5, 10.0), 
        OddsFilterStrategy(5, 20.0),
        OddsFilterStrategy(4, 10.0),
        OddsFilterStrategy(4, 15.0),
    ]
    
    summary_data = []
    
    for strategy in strategies:
        print(f"Simulating {strategy.name}...")
        total_cost = 0
        total_return = 0
        races_hit = 0
        
        for race_id in race_ids:
            # そのレースの予測データ
            df_race = df_all[df_all['race_id'] == race_id].copy()
            
            bets = strategy.decide_bets(df_race)
            
            # 結果データ
            race_result = None
            str_rid = str(race_id)
            if str_rid in results_db.index:
                race_result = results_db.loc[str_rid]
            elif hasattr(results_db.index, 'levels') and str_rid in results_db.index.get_level_values(0):
                 race_result = results_db.loc[str_rid]
                 
            for bet in bets:
                cost = bet['cost']
                payout, is_hit = calculate_payout_any(bet, race_result)
                
                total_cost += cost
                total_return += payout
                if is_hit:
                    races_hit += 1 # 1レースで複数的中（フォーメーション等）は考慮しない単純計算
        
        profit = total_return - total_cost
        roi = (total_return / total_cost * 100) if total_cost > 0 else 0
        hit_rate = (races_hit / len(race_ids) * 100)
        
        summary_data.append({
            'Strategy': strategy.name,
            'Cost': total_cost,
            'Return': total_return,
            'Profit': profit,
            'ROI': roi,
            'HitRate': hit_rate
        })

    df_summary = pd.DataFrame(summary_data).sort_values('Profit', ascending=False)
    print("\nSimulation Results:")
    print(df_summary.to_markdown(index=False, floatfmt=".1f"))
    
    # レポート保存
    report_path = "reports/simulation_ltr_20260214.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LTR戦略シミュレーション (2026/02/14)\n\n")
        f.write(df_summary.to_markdown(index=False, floatfmt=".1f"))
    print(f"Report saved: {report_path}")

if __name__ == "__main__":
    main()
