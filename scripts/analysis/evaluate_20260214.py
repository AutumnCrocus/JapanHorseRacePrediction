import sys
import os
import re
import pandas as pd
from datetime import datetime

# モジュールパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import requests
from bs4 import BeautifulSoup
from io import StringIO
from modules.scraping import Return  # noqa: E402
from modules.constants import HEADERS

# モンキーパッチ: 当日レース結果取得用
def scrape_single_return_live(race_id, session):
    """当日レース結果（速報）を取得"""
    # https://race.netkeiba.com/race/result.html?race_id=2024...
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        response.encoding = "EUC-JP"
        
        if response.status_code != 200:
            return race_id, None

        soup = BeautifulSoup(response.text, "lxml")
        
        # race.netkeiba.com の払い戻しエリア
        # PC版: <div class="Result_Pay_Back"> or <div class="PayBackHost">
        # SP版: <div class="RaceResult_Return">
        payback_div = soup.find("div", {"class": "PayBackHost"}) or \
                      soup.find("div", {"class": "RaceResult_Return"}) or \
                      soup.find("div", {"class": "Full_PayBack_List"}) or \
                      soup.find("table", {"class": "PayBack_Table"}) # テーブル直接検索も追加

        if payback_div:
            # テーブルが複数ある場合がある（単勝など、3連複など）
            dfs = pd.read_html(StringIO(str(payback_div)))
            if dfs:
                merged_df = pd.concat(dfs)
                return race_id, merged_df
        
        # テーブルが見つからない場合、ページ全体のテーブルから探す（強引）
        dfs = pd.read_html(StringIO(response.text))
        for df in dfs:
            if "3連複" in str(df.values) or "３連複" in str(df.values):
                return race_id, df

        return race_id, None
    except Exception as e:
        # print(f"Error scraping {race_id}: {e}")
        return race_id, None

# パッチ適用
Return._scrape_single_return = scrape_single_return_live


# レースID定義 (2026/02/14)
RACE_IDS = []
# 東京 (05) 1回5日目, 京都 (08) 2回5日目, 小倉 (10) 1回7日目
RACE_IDS += [f"2026050105{r:02}" for r in range(1, 13)]
RACE_IDS += [f"2026080205{r:02}" for r in range(1, 13)]
RACE_IDS += [f"2026100107{r:02}" for r in range(1, 13)]

REPORTS = {
    'Box4 (LGBM)': r'reports/prediction_20260214_box4_sanrenpuku.md',
    'Box5 (LGBM)': r'reports/prediction_20260214_box5_sanrenpuku.md',
    'Box5 (LTR)': r'reports/prediction_20260214_ltr_box5.md'
}


def parse_prediction_report(filepath):  # noqa: C901
    """
    レポートMarkdownから推奨買い目を抽出する。
    戻り値: {race_id: {'type': 'box', 'horses': [1,2,3...], 'cost': int}}
    """
    bets = {}

    if not os.path.exists(filepath):
        print(f"Warning: Report file not found: {filepath}")
        return bets

    current_venue = None
    current_race_num = None

    # 会場名と場所コードの対応
    venue_map = {'東京': '05', '京都': '08', '小倉': '10'}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 会場検出: "## 東京開催"
            m_venue = re.match(r'^##\s+(.+)開催', line)
            if m_venue:
                current_venue = m_venue.group(1)
                continue

            # レース番号検出: "### 東京01R" or "### 東京1R"
            m_race = re.match(r'^###\s+(.+?)(\d{1,2})R', line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))

                if current_venue and venue_name != current_venue:
                    current_venue = venue_name

                current_race_num = race_num
                continue

            # 買い目抽出
            m_bet = re.match(r'-\s+\*\*3連複\s+BOX\*\*:\s+([\d,]+)\s+BOX\s+\((\d+)円\)', line)
            if m_bet and current_venue and current_race_num:
                horses_str = m_bet.group(1)
                cost = int(m_bet.group(2))
                horses = [int(x) for x in horses_str.split(',')]

                # レースIDを特定
                place_code = venue_map.get(current_venue)
                if place_code:
                    target_id = None
                    for rid in RACE_IDS:
                        # rid[4:6] == place_code, rid[10:12] == race_num
                        if rid[4:6] == place_code and int(rid[10:12]) == current_race_num:
                            target_id = rid
                            break

                    if target_id:
                        bets[target_id] = {
                            'type': 'box_sanrenpuku',
                            'horses': horses,
                            'cost': cost
                        }
                    else:
                        print(f"Warning: Race ID not found for {current_venue} {current_race_num}R")

    return bets


def calculate_payout(bet_info, result_df):
    """
    3連複BOXの払戻金を計算する
    bet_info: {'horses': [1,2,3,4,5], 'cost': 1000}
    result_df: 払戻データのDataFrame
    """
    if result_df is None or result_df.empty:
        return 0, False

    total_return = 0
    hit = False

    try:
        # col 0: 券種, col 1: 馬番, col 2: 払戻
        # カラム名ではなく位置で参照する
        #print(f"DEBUG: Processing race result with {len(result_df)} rows")
        for _, row in result_df.iterrows():
            #print(f"DEBUG ROW: {row.values}")
            val0 = str(row.iloc[0])
            if '3連複' not in val0:
                continue

            # 馬番: "9 - 11 - 14"
            umaban_str = str(row.iloc[1])
            winning_horses = [int(x) for x in re.findall(r'\d+', umaban_str)]

            # 払戻金: "12,340"
            payout_str = str(row.iloc[2])
            payout = int(re.sub(r'[^\d]', '', payout_str))

            print(f"DEBUG HIT CHECK: Winning={winning_horses}, Bet={bet_info['horses']}")
            
            # 的中判定
            bet_horses = set(bet_info['horses'])
            if set(winning_horses).issubset(bet_horses):
                print(f"DEBUG HIT! Payout={payout}")
                total_return += payout
                hit = True

    except Exception as e:
        print(f"Error calculation: {e}")

    return total_return, hit


def main():  # noqa: C901
    print("Fetching race results...")
    results = Return.scrape(RACE_IDS)
    print(f"DEBUG: Results shape: {results.shape}")
    if not results.empty:
        print(f"DEBUG: Results head:\n{results.head()}")
        print(f"DEBUG: Results index levels: {results.index.nlevels}")
        # インデックスのサンプル確認
        print(f"DEBUG: Results index sample: {results.index[:5]}")

    summary = []

    for name, report_path in REPORTS.items():
        print(f"Evaluating {name}...")
        bets = parse_prediction_report(report_path)

        total_cost = 0
        total_return = 0
        races_bet = 0
        races_hit = 0

        for race_id in RACE_IDS:
            if race_id not in bets:
                continue

            bet = bets[race_id]
            cost = bet['cost']

            # 結果データの抽出 MultiIndex対応
            race_result = None
            # resultsのindexにrace_idが含まれているか確認
            if race_id in results.index:
                # MultiIndexの場合、loc[race_id]で該当レースの全行が取れる
                race_result = results.loc[race_id]
            elif hasattr(results.index, 'levels') and race_id in results.index.get_level_values(0):
                # 明示的にレベル指定で確認
                race_result = results.loc[race_id]

            payout, is_hit = calculate_payout(bet, race_result)

            total_cost += cost
            total_return += payout
            races_bet += 1
            if is_hit:
                races_hit += 1

        profit = total_return - total_cost
        roi = (total_return / total_cost * 100) if total_cost > 0 else 0
        hit_rate = (races_hit / races_bet * 100) if races_bet > 0 else 0

        summary.append({
            'Strategy': name,
            'Total Return': total_return,
            'Total Cost': total_cost,
            'Net Profit': profit,
            'ROI': roi,
            'Hit Rate': hit_rate,
            'Bets': races_bet,
            'Hits': races_hit
        })

    # レポート出力
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values('Net Profit', ascending=False)

    report_file = 'reports/comparison_20260214.md'
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 2026/02/14 予測成績比較\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 概要\n\n")
        f.write(df_summary.to_markdown(index=False, floatfmt=".1f"))
        f.write("\n\n")

        f.write("## 詳細分析\n\n")
        for _, row in df_summary.iterrows():
            f.write(f"### {row['Strategy']}\n")
            f.write(f"- 投資総額: {row['Total Cost']:,}円\n")
            f.write(f"- 回収総額: {row['Total Return']:,}円\n")
            f.write(f"- 純利益: {row['Net Profit']:+,}円\n")
            f.write(f"- 回収率: {row['ROI']:.1f}%\n")
            f.write(f"- 的中率: {row['Hit Rate']:.1f}% ({row['Hits']}/{row['Bets']})\n\n")

    print(f"Comparison report generated: {report_file}")
    print(df_summary)


if __name__ == "__main__":
    main()
