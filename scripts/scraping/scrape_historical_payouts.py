
import os
import sys
import pickle
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import argparse

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS

# ============= 設定 =============
START_YEAR = 2010
END_YEAR = 2026
BASE_OUTPUT_DIR = "data/raw/payouts"
MAX_WORKERS = 10 # サーバー負荷を考慮して調整
# ===============================

def get_race_id_list(year):
    """
    指定年の全レースIDリストを生成する
    形式: YYYY PP KK DD RR
    """
    race_ids = []
    # 場所コード: 01(札幌)〜10(小倉)
    places = range(1, 11)
    # 回数: 1〜6回 (目安)
    kais = range(1, 7)
    # 日数: 1〜12日 (目安)
    days = range(1, 13)
    # レース: 1〜12R
    races = range(1, 13)
    
    # 効率化のため、存在する可能性が高い範囲に絞ることも可能だが、
    # 網羅性を優先して総当たりリストを生成し、存在確認はスクレイピング時に行う
    # ただし、404エラーが多すぎると遅くなるため、ある程度絞るのが理想
    # ここでは簡易的に全組み合わせ生成 (約 10*6*12*12 = 8640通り)
    
    for p in places:
        for k in kais:
            for d in days:
                for r in races:
                    rid = f"{year}{p:02}{k:02}{d:02}{r:02}"
                    race_ids.append(rid)
    
    return race_ids

def parse_horse_numbers(umaban_str):
    """
    "1 - 2" や "1" などの文字列をタプル (1, 2) や (1,) に変換する。
    " → " 区切りにも対応。
    """
    clean_str = umaban_str.replace(' - ', '-').replace(' → ', '-').replace(' ', '-')
    nums = re.findall(r'\d+', clean_str)
    if nums:
        return tuple(int(n) for n in nums)
    return None

def scrape_payout_data_from_html(race_id):
    """
    race.netkeiba.com の結果ページから配当データを取得する
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        # タイムアウトを少し長めに
        res = requests.get(url, headers=HEADERS, timeout=10)
        
        # 404や存在しないレースの場合
        if res.status_code != 200:
            return None
            
        soup = BeautifulSoup(res.content, 'lxml', from_encoding='euc-jp')
        
        # タイトルで「開催データがありません」などをチェック
        if "開催データがありません" in soup.text:
            return None

        # pandasでテーブル取得
        try:
            from io import StringIO
            dfs = pd.read_html(StringIO(str(soup)))
        except ValueError:
            return None
        
        target_dfs = []
        keywords = ["単勝", "単 勝", "複勝", "複 勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "三連複", "3連単", "三連単"]
        
        for df in dfs:
            df_str = df.to_string()
            if any(k in df_str for k in keywords):
                target_dfs.append(df)
        
        if not target_dfs:
            return None
            
        race_payouts = {}
        
        for df in target_dfs:
            current_type = None
            
            for idx, row in df.iterrows():
                row_list = [str(x) for x in row.tolist()]
                row_list = [x for x in row_list if x != 'nan']
                if not row_list: continue
                
                label = row_list[0]
                
                if "単勝" in label or "単 勝" in label: current_type = 'tan'
                elif "複勝" in label or "複 勝" in label: current_type = 'fuku'
                elif "枠連" in label: current_type = 'wakuren'
                elif "馬連" in label: current_type = 'umaren'
                elif "ワイド" in label: current_type = 'wide'
                elif "馬単" in label: current_type = 'umatan'
                elif "3連複" in label or "三連複" in label: current_type = 'sanrenpuku'
                elif "3連単" in label or "三連単" in label: current_type = 'sanrentan'
                
                if current_type is None: continue
                
                if len(row_list) > 2:
                    try:
                        umaban_raw = str(row_list[1])
                        payout_raw = str(row_list[2]).replace(',', '').replace('円', '')
                        
                        p_splits = payout_raw.split()
                        
                        is_multi = (current_type in ['fuku', 'wide']) and (len(p_splits) > 1)

                        if is_multi:
                            all_nums = re.findall(r'\d+', umaban_raw)
                            if current_type == 'fuku':
                                if len(all_nums) == len(p_splits):
                                    for i, price_str in enumerate(p_splits):
                                        if not price_str.isdigit(): continue
                                        key = int(all_nums[i])
                                        price = int(price_str)
                                        if current_type not in race_payouts: race_payouts[current_type] = {}
                                        race_payouts[current_type][key] = price
                            elif current_type == 'wide':
                                if len(all_nums) == len(p_splits) * 2:
                                    for i, price_str in enumerate(p_splits):
                                        if not price_str.isdigit(): continue
                                        h1 = int(all_nums[i*2])
                                        h2 = int(all_nums[i*2+1])
                                        key = tuple(sorted((h1, h2)))
                                        price = int(price_str)
                                        if current_type not in race_payouts: race_payouts[current_type] = {}
                                        race_payouts[current_type][key] = price
                        
                        else:
                            if not payout_raw.isdigit(): continue
                            payout = int(payout_raw)
                            
                            key = None
                            if current_type == 'tan':
                                key = int(umaban_raw)
                            elif current_type == 'fuku':
                                key = int(umaban_raw)
                            else:
                                nums = re.findall(r'\d+', umaban_raw)
                                if nums:
                                    if current_type in ['umaren', 'wide', 'sanrenpuku']:
                                        key = tuple(sorted([int(n) for n in nums]))
                                    else:
                                        key = tuple([int(n) for n in nums])
                            
                            if not key: continue
                            
                            if current_type not in race_payouts: race_payouts[current_type] = {}
                            race_payouts[current_type][key] = payout
                        
                    except Exception:
                        pass
                        
        return race_payouts

    except Exception:
        return None

def process_year(year):
    print(f"\nProcessing Year: {year}")
    
    # 保存ディレクトリ作成
    year_dir = os.path.join(BASE_OUTPUT_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    output_file = os.path.join(year_dir, f"payouts_{year}.pkl")
    
    # 既存データの読み込み
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                existing_data = pickle.load(f)
            print(f"  Loaded {len(existing_data)} existing races.")
        except:
            print("  Failed to load existing data. Starting fresh.")
    
    # IDリスト生成
    all_ids = get_race_id_list(year)
    
    # スクレイピング対象（未取得のもの、または全更新なら全て）
    # 今回は「再取得」のリクエストなので、基本的には全チェックを行いたいが、
    # 効率のため既存データがあるIDはスキップするオプションも考慮
    # -> ユーザー要望は「再取得」なので、既存にあっても上書きチェックするほうが安全だが、
    # 過去データは変わらないので、存在すればスキップで良いはず。
    # ただし今回は不完全なデータ補完の目的もあるかもしれないので、
    # 簡易的に「全IDを対象」とし、取得できた場合のみupdateする
    
    # 効率化: 既にデータがあるIDはスキップし、新規IDのみ取得する
    # to_scrape = [rid for rid in all_ids if rid not in existing_data]
    
    # 「全て再取得」という指示なので、全リストを対象にする
    to_scrape = all_ids
    
    print(f"  Target: {len(to_scrape)} potential IDs.")
    
    results = existing_data.copy()
    new_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdmで進捗表示
        future_to_id = {executor.submit(scrape_payout_data_from_html, rid): rid for rid in to_scrape}
        
        for future in tqdm(as_completed(future_to_id), total=len(to_scrape), desc=f"Scraping {year}"):
            rid = future_to_id[future]
            try:
                data = future.result()
                if data:
                    results[rid] = data
                    new_count += 1
            except Exception as e:
                # print(f"Error {rid}: {e}")
                pass
                
    # 保存
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"  Saved {len(results)} races to {output_file} (New/Updated: {new_count})")

def main():
    parser = argparse.ArgumentParser(description="Scrape historical payout data.")
    parser.add_argument("--start", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end", type=int, default=END_YEAR, help="End year")
    parser.add_argument("--force", action="store_true", help="Force re-scrape all")
    args = parser.parse_args()
    
    print("=== Historical Payout Scraping Start ===")
    print(f"Years: {args.start} - {args.end}")
    
    for year in range(args.start, args.end + 1):
        process_year(year)
        
    print("=== All Completed ===")

if __name__ == "__main__":
    main()
