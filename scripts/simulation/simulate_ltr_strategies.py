import sys
import os
import re
import itertools
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

# モジュールパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

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
                # DEBUG
                # print(f"DEBUG: Scraped DF shape: {merged_df.shape}")
                # print(f"DEBUG: Scraped DF head: {merged_df.head().to_string()}")
                return race_id, merged_df
        
        dfs = pd.read_html(StringIO(response.text))
        valid_dfs = []
        for df in dfs:
            # 払い戻しに関連するキーワードが含まれているかチェック
            s = str(df.values)
            if any(k in s for k in ["単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単", "３連複", "３連単"]):
                valid_dfs.append(df)
        
        if valid_dfs:
            return race_id, pd.concat(valid_dfs)
        
        return race_id, None
    except Exception as e:
        print(f"DEBUG: Scrape Error: {e}")
        return race_id, None

# Return.scrapeの実装依存だが、通常は辞書やリストで集めてconcatする。
# ここで Return.scrape をモンキーパッチする方が確実かもしれない。
# しかし、まずは scrape_single_return_live の戻り値が (race_id, df) であることは正しいはず。
# 問題は Return.scrape がどう結合しているか。
# simulate_ltr_strategies.py の main関数で results_db = Return.scrape(race_id_list) している。
# 返り値 results_db のインデックスが race_id になっていることを期待している。
# もしなっていないなら、ここで強制的に Return.scrape もモンキーパッチする。

def scrape_batch(race_ids):
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # requests.Session() は thread-safe ではないので都度作成か、あるいは各スレッドで
        pass 
    
    # 簡易実装: 直列実行で確実にする
    dfs = []
    import requests
    session = requests.Session()
    from tqdm import tqdm
    
    print(f"DEBUG: Scraping {len(race_ids)} races directly...")
    for rid in tqdm(race_ids):
        _, df = scrape_single_return_live(rid, session)
        if df is not None:
            df['race_id'] = str(rid) # カラムに追加
            df.set_index('race_id', inplace=True) # インデックスに設定
            dfs.append(df)
    
    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()

Return.scrape = scrape_batch
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
    def __init__(self, bet_type, r1, r2, r3=None):
        """
        bet_type: 'sanrenpuku' or 'sanrentan'
        r1, r2, r3: list of ranks (1-indexed) e.g. [1, 2, 3]
        """
        self.bet_type = bet_type
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        name = f"Fm({bet_type}) {r1}-{r2}" + (f"-{r3}" if r3 else "")
        super().__init__(name)

    def decide_bets(self, df_race):
        df = df_race.sort_values('ltr_score', ascending=False)
        # ランクごとの馬番マップ (1-indexed)
        rank_horse = {}
        for i in range(len(df)):
            horse_num = df.iloc[i]['horse_number']
            rank_horse[i+1] = horse_num
        
        # 該当する馬番を取得 (存在しないランクは無視)
        horses1 = [rank_horse[r] for r in self.r1 if r in rank_horse]
        horses2 = [rank_horse[r] for r in self.r2 if r in rank_horse]
        horses3 = []
        if self.r3:
            horses3 = [rank_horse[r] for r in self.r3 if r in rank_horse]
            
        combinations = []
        
        if self.bet_type == 'sanrenpuku':
            # 3連複フォーメーション (順序関係なし、重複なし)
            if not self.r3:
                horses3 = horses2
                
            for h1 in horses1:
                for h2 in horses2:
                    for h3 in horses3:
                        if h1 != h2 and h1 != h3 and h2 != h3:
                            combo = tuple(sorted([h1, h2, h3]))
                            combinations.append(combo)
                            
        elif self.bet_type == 'sanrentan':
            # 3連単フォーメーション (順序あり)
            if not self.r3:
                 horses3 = horses2 # フォールバック
                 
            for h1 in horses1:
                for h2 in horses2:
                    for h3 in horses3:
                        if h1 != h2 and h1 != h3 and h2 != h3:
                            combo = (h1, h2, h3)
                            combinations.append(combo)

        # 重複排除
        unique_combos = sorted(list(set(combinations)))
        cost = len(unique_combos) * 100
        
        return [{'type': self.bet_type, 'combinations': unique_combos, 'cost': cost}]


class WinStrategy(Strategy):
    def __init__(self, prob_threshold=0.0, odds_threshold=0.0, ev_threshold=0.0):
        name = f"Win(P>{prob_threshold}, O>{odds_threshold}, EV>{ev_threshold})"
        super().__init__(name)
        self.prob_threshold = prob_threshold
        self.odds_threshold = odds_threshold
        self.ev_threshold = ev_threshold
        
    def decide_bets(self, df_race):
        df = df_race.sort_values('ltr_score', ascending=False)
        # 1位の馬を対象
        top = df.iloc[0]
        
        prob = top['probability']
        odds = top['win_odds']
        ev = prob * odds
        
        if prob >= self.prob_threshold and odds >= self.odds_threshold and ev >= self.ev_threshold:
            return [{'type': 'win', 'horse': top['horse_number'], 'cost': 100}]
        
        return []

class UmarenStrategy(Strategy):
    def __init__(self, method='box', n_horses=5, n_opponents=5):
        if method == 'box':
            name = f"Umaren Box({n_horses})"
        else:
            name = f"Umaren Nagashi(1-{n_opponents})"
        super().__init__(name)
        self.method = method
        self.n_horses = n_horses
        self.n_opponents = n_opponents
        
    def decide_bets(self, df_race):
        df = df_race.sort_values('ltr_score', ascending=False)
        top = df['horse_number'].tolist()
        
        combinations = []
        if self.method == 'box':
            horses = top[:self.n_horses]
            import itertools
            for h1, h2 in itertools.combinations(horses, 2):
                combinations.append(tuple(sorted((h1, h2))))
        else: # nagashi
            axis = top[0]
            opponents = top[1:self.n_opponents+1]
            for opp in opponents:
                combinations.append(tuple(sorted((axis, opp))))
                
        cost = len(combinations) * 100
        return [{'type': 'umaren', 'combinations': combinations, 'cost': cost}]

def calculate_payout_any(bet_info, result_df):
    """汎用払い戻し計算"""
    if result_df is None or result_df.empty:
        return 0, False
        
    total_return = 0
    hit = False
    
    try:
        # 共通処理: 行ループ
        for i, row in result_df.iterrows():
            # print(f"DEBUG: Row {i}: {row.values}") # 全行出力
            val0 = str(row.iloc[0])
            # print(f"DEBUG: Row val0='{val0}'") # 超詳細ログ
            
            # 単勝
            if bet_info['type'] == 'win':
                if '単勝' in val0:
                    umaban_str = str(row.iloc[1])
                    payout_str = str(row.iloc[2])
                    # print(f"DEBUG: Win Row Found. Umaban='{umaban_str}' Payout='{payout_str}' BetHorse={bet_info['horse']}")
                    try:
                        winning_horse = int(re.sub(r'[^\d]', '', umaban_str))
                        if winning_horse == bet_info['horse']:
                            payout = int(re.sub(r'[^\d]', '', payout_str))
                            total_return += payout
                            hit = True
                            # print(f"DEBUG: Win HIT! {payout} yen")
                    except:
                        pass
            
            # 馬連
            elif bet_info['type'] == 'umaren':
                if '馬連' in val0:
                    umaban_str = str(row.iloc[1])
                    payout_str = str(row.iloc[2])
                    # print(f"DEBUG: Umaren Row Found. Umaban='{umaban_str}'")
                    nums = [int(x) for x in re.findall(r'\d+', umaban_str)]
                    if len(nums) == 2:
                        result_combo = tuple(sorted(nums))
                        if result_combo in bet_info['combinations']:
                             payout = int(re.sub(r'[^\d]', '', payout_str))
                             total_return += payout
                             hit = True
                             # print(f"DEBUG: Umaren HIT! {payout} yen")

            # 3連複BOX
            elif bet_info['type'] == 'box_sanrenpuku':
                 if '3連複' in val0 or '３連複' in val0:
                    bet_horses = set(bet_info['horses'])
                    if len(bet_horses) < 3: continue
                    
                    umaban_str = str(row.iloc[1])
                    winning_horses = [int(x) for x in re.findall(r'\d+', umaban_str)]
                    if set(winning_horses).issubset(bet_horses):
                        payout = int(re.sub(r'[^\d]', '', str(row.iloc[2])))
                        total_return += payout
                        hit = True

            # フォーメーション (3連複/3連単)
            elif bet_info['type'] in ['sanrenpuku', 'sanrentan']:
                target_str = '3連複' if bet_info['type'] == 'sanrenpuku' else '3連単'
                target_str_alt = '３連複' if bet_info['type'] == 'sanrenpuku' else '３連単'
                
                if target_str in val0 or target_str_alt in val0:
                    umaban_str = str(row.iloc[1])
                    nums = [int(x) for x in re.findall(r'\d+', umaban_str)]
                    
                    if bet_info['type'] == 'sanrenpuku':
                        if len(nums) == 3:
                            result_combo = tuple(sorted(nums))
                            if result_combo in bet_info['combinations']:
                                payout = int(re.sub(r'[^\d]', '', str(row.iloc[2])))
                                total_return += payout
                                hit = True
                    else: # sanrentan
                        if len(nums) == 3:
                            result_combo = tuple(nums)
                            if result_combo in bet_info['combinations']:
                                payout = int(re.sub(r'[^\d]', '', str(row.iloc[2])))
                                total_return += payout
                                hit = True

    except Exception as e:
         print(f"Error in calculate_payout_any: {e}")

    except Exception:
         pass
        
    return total_return, hit


def main():  # noqa: C901
    # データ読み込み
    csv_path = "data/processed/prediction_202412_ltr.csv"
    if not os.path.exists(csv_path):
        print("Error: Prediction data not found. Run predict script first.")
        return
        
    df_all = pd.read_csv(csv_path)
    # probability カラムがない場合、ltr_scoreから計算 (Softmax)
    if 'probability' not in df_all.columns:
        import numpy as np
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
            
        print("Calculating probabilities from LTR scores...")
        df_all['probability'] = df_all.groupby('race_id')['ltr_score'].transform(softmax)
        
    race_ids = df_all['race_id'].unique()
    print(f"Loaded prediction data for {len(race_ids)} races.")
    
    # 払い戻しデータ取得
    print("Fetching race results...")
    race_id_list = [str(rid) for rid in race_ids]
    results_db = Return.scrape(race_id_list)
    
    # DEBUG: ID確認
    # print(f"DEBUG: CSV Race IDs (sample): {race_ids[:5]}")
    # print(f"DEBUG: Results DB Index (sample): {results_db.index[:5]}")
    
    # 戦略定義
    strategies = [
        # 単勝戦略 (控除率80%の恩恵)
        WinStrategy(prob_threshold=0.0, odds_threshold=0.0, ev_threshold=0.0), # 単勝全買い(1位)
        WinStrategy(prob_threshold=0.3, odds_threshold=2.0), # 信頼度・オッズフィルター
        WinStrategy(ev_threshold=1.2), # 期待値フィルター(1.2)
        WinStrategy(ev_threshold=1.5), # 期待値フィルター(1.5)
        
        # 馬連戦略 (控除率77.5%)
        UmarenStrategy('box', n_horses=5), # 5頭BOX (10点)
        UmarenStrategy('nagashi', n_opponents=4), # 1頭軸-4頭流し (4点)
        UmarenStrategy('nagashi', n_opponents=5), # 1頭軸-5頭流し (5点)

        # 既存戦略
        BoxStrategy(3),
        BoxStrategy(4),
        FormationStrategy('sanrenpuku', [1], [2,3,4,5,6], [2,3,4,5,6]), 
        FormationStrategy('sanrentan', [1,2,3], [1,2,3], [1,2,3]),
    ]
    
    summary_data = []
    
    for strategy in strategies:
        print(f"Simulating {strategy.name}...")
        total_cost = 0
        total_return = 0
        total_profit = 0
        races_hit = 0
        bet_count = 0
        
        for race_id in race_ids:
            df_race = df_all[df_all['race_id'] == race_id].copy()
            bets = strategy.decide_bets(df_race)
            
            # DEBUG
            if len(bets) > 0 and bets[0].get('cost', 0) > 0:
                pass
                # print(f"DEBUG: {strategy.name} Race:{race_id} Bets:{len(bets)} Cost:{bets[0]['cost']}")
            
            # 結果データ
            race_result = None
            str_rid = str(race_id)
            if str_rid in results_db.index:
                race_result = results_db.loc[str_rid]
            elif hasattr(results_db.index, 'levels') and str_rid in results_db.index.get_level_values(0):
                 race_result = results_db.loc[str_rid]
            
            # DEBUG
            if race_result is None or race_result.empty:
                # print(f"DEBUG: No result data for Race:{race_id}")
                pass
            else:
                # print(f"DEBUG: Result data found for Race:{race_id}. Shape:{race_result.shape}")
                pass
                 
            for bet in bets:
                cost = bet['cost']
                if cost == 0: continue
                
                bet_count += 1
                # print(f"DEBUG: Calculating payout for {bet['type']}...")
                payout, is_hit = calculate_payout_any(bet, race_result)
                
                total_cost += cost
                total_return += payout
                if is_hit:
                    races_hit += 1 
        
        profit = total_return - total_cost
        roi = (total_return / total_cost * 100) if total_cost > 0 else 0
        hit_rate = (races_hit / len(race_ids) * 100)
        
        # フィルター系はベットしないレースがあるので、実質的中率(ベットしたレース中の的中率)も重要だが
        # ここでは全体の的中率を表示する
        
        summary_data.append({
            'Strategy': strategy.name,
            'Cost': total_cost,
            'Return': total_return,
            'Profit': profit,
            'ROI': roi,
            'HitRate': hit_rate,
            'BetCount': bet_count
        })

    df_summary = pd.DataFrame(summary_data).sort_values('ROI', ascending=False)
    print("\nSimulation Results:")
    print(df_summary.to_markdown(index=False, floatfmt=".1f"))
    
    # レポート保存
    report_path = "reports/simulation_ltr_20260214_v2.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LTR戦略シミュレーション (2026/02/14) - 単勝・馬連・高期待値フィルター\n\n")
        f.write(df_summary.to_markdown(index=False, floatfmt=".1f"))
    print(f"Report saved: {report_path}")

if __name__ == "__main__":
    main()
