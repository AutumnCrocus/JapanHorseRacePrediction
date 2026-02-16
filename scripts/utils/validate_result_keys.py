
import sys
import os
import pandas as pd

# モジュールパス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.scraping import Return
from modules.constants import HEADERS

# モンキーパッチ適用 (シミュレーションスクリプトと同様)
from io import StringIO
from bs4 import BeautifulSoup
import requests

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

def main():
    csv_path = "data/processed/prediction_20260214_ltr_full.csv"
    if not os.path.exists(csv_path):
        print("Error: Prediction data not found.")
        return
        
    df_all = pd.read_csv(csv_path)
    race_ids = df_all['race_id'].unique()
    print(f"CSV Race IDs (First 5): {race_ids[:5]}")
    print(f"CSV Race ID Type: {type(race_ids[0])}")
    
    # 文字列変換して取得
    race_id_list = [str(rid) for rid in race_ids[:3]] # 3レースだけテスト
    print(f"Fetching results for: {race_id_list}")
    
    results_db = Return.scrape(race_id_list)
    
    print("\nResults DB Index:")
    print(results_db.index)
    if hasattr(results_db.index, 'levels'):
        print(f"MultiIndex Levels: {results_db.index.levels}")
    
    print("\nMatching Check:")
    for rid in race_ids[:3]:
        str_rid = str(rid)
        is_in = str_rid in results_db.index
        print(f"RaceID {rid} (str={str_rid}) in Index? -> {is_in}")
        if not is_in and hasattr(results_db.index, 'levels'):
             is_in_level = str_rid in results_db.index.get_level_values(0)
             print(f"  In Level 0? -> {is_in_level}")

if __name__ == "__main__":
    main()
