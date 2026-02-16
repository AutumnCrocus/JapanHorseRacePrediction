
import os
import sys
import pickle
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS

# ============= 設定 =============
TARGET_DATE = '20260208' # 過去の確定済みレース日
OUTPUT_FILE = f"data/raw/payouts_verify_{TARGET_DATE}.pkl"
# ===============================

def scan_race_ids_brute_force(date_str):
    """
    総当たりでその日に開催されるレースIDを特定する
    ID形式: YYYY PP KK DD RR
    """
    
    year = date_str[:4]
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    target_date_jp = f"{year}年{month}月{day}日"
    print(f"Target Date: {target_date_jp}")
    
    # 探索範囲 (東京05, 京都08, 小倉10) - 2026/02/08の開催場所を想定
    # 2月なので東京(05), 京都(08), 小倉(10) がメインだが、念のため全場チェックも可能だが遅くなる
    # ここでは既存ロジックを踏襲
    places = [5, 8, 10] 
    kais = range(1, 6)    # 回数
    days = range(1, 13)   # 日目
    
    keys = []
    for p in places:
        for k in kais:
            for d in days:
                keys.append(f"{year}{p:02}{k:02}{d:02}")
    
    active_keys = []
    
    def check_key(key):
        # 1RのIDで存在確認
        rid = key + "01"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.encoding = 'EUC-JP'
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    print(f"DEBUG: Found match {rid}")
                    return key
            return None
        except:
            return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in as_completed(futures):
            result = future.result()
            if result:
                active_keys.append(result)
                
    final_ids = []
    for key in sorted(active_keys):
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
            
    print(f"Brute-force scan found {len(final_ids)} races.")
    return final_ids

def get_race_ids(date_str):
    """指定された日付のレースIDを取得する"""
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    print(f"Fetching race IDs from: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = 'EUC-JP'
        soup = BeautifulSoup(response.text, 'lxml')
        links = soup.find_all('a', href=True)
        race_ids = []
        for link in links:
            href = link['href']
            match = re.search(r'race_id=(\d+)', href)
            if match:
                rid = match.group(1)
                # target_dateのレースIDだけ抽出（URLには前後の日付も含まれることがあるため）
                # ただしrace_listは通常その日のものだけだが、念のため
                if rid.startswith(date_str[:4]): 
                    race_ids.append(rid)
        
        race_ids = sorted(list(set(race_ids)))
        print(f"Found {len(race_ids)} races.")
        
        if not race_ids:
            print("Trying brute-force scan...")
            race_ids = scan_race_ids_brute_force(date_str)
            
        return race_ids
    except Exception as e:
        print(f"Error fetching race IDs: {e}")
        # フォールバック
        return scan_race_ids_brute_force(date_str)

def parse_horse_numbers(umaban_str):
    """
    "1 - 2" や "1" などの文字列をタプル (1, 2) や (1,) に変換する。
    " → " 区切りにも対応。
    """
    # 余計な空白や記号を整理
    clean_str = umaban_str.replace(' - ', '-').replace(' → ', '-').replace(' ', '-')
    
    # 複勝などで "2 7 14" のようにスペース区切りでくるケースに対応
    # 上記replaceで "2-7-14" になっている可能性があるが、
    # 元が "2 7 14" (複勝) なのか "2 - 7" (馬連) なのか
    
    import re
    nums = re.findall(r'\d+', umaban_str)
    if nums:
        return tuple(int(n) for n in nums)
    return None

def scrape_payout_data_from_html(race_id):
    """
    race.netkeiba.com の結果ページから配当データを取得する
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    # print(f"Fetching: {url}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.content, 'lxml', from_encoding='euc-jp')
        
        try:
            from io import StringIO
            dfs = pd.read_html(StringIO(str(soup)))
        except ValueError:
            print(f"No tables found in {race_id}")
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
                
                # 券種切り替え
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
                        
                        # 複勝・ワイドの複数的中対応
                        is_multi = (current_type in ['fuku', 'wide']) and (len(p_splits) > 1)

                        if is_multi:
                            # 馬番も分割する必要がある
                            # 複勝: "2 7 14" -> [2, 7, 14]
                            # ワイド: "2 - 7 2 - 14 7 - 14" -> [(2,7), (2,14), (7,14)]
                            # ただしHTMLの構造上、改行タグなどがスペースに変換されていると仮定
                            
                            # 馬番文字列から数値を全て抽出
                            all_nums = re.findall(r'\d+', umaban_raw)
                            
                            if current_type == 'fuku':
                                # 複勝: 配当数 = 馬番数
                                if len(all_nums) == len(p_splits):
                                    for i, price_str in enumerate(p_splits):
                                        if not price_str.isdigit(): continue
                                        key = int(all_nums[i])
                                        price = int(price_str)
                                        if current_type not in race_payouts: race_payouts[current_type] = {}
                                        race_payouts[current_type][key] = price
                                else:
                                    print(f"WARNING: Indeterminate Fuku format {race_id}: {umaban_raw} vs {payout_raw}")

                            elif current_type == 'wide':
                                # ワイド: 配当数 * 2 = 馬番数 (ペアなので)
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
                                     print(f"WARNING: Indeterminate Wide format {race_id}: {umaban_raw} vs {payout_raw}")
                        
                        else:
                            # 単一配当（または分離行で来ている場合）
                            if not payout_raw.isdigit(): continue
                            payout = int(payout_raw)
                            
                            key = None
                            if current_type == 'tan':
                                key = int(umaban_raw)
                            elif current_type == 'fuku':
                                # たまに複勝が1行ずつ分かれていることもあるかも？
                                key = int(umaban_raw)
                            else:
                                # 連勝系
                                nums = re.findall(r'\d+', umaban_raw)
                                if nums:
                                    if current_type in ['umaren', 'wide', 'sanrenpuku']:
                                        key = tuple(sorted([int(n) for n in nums]))
                                    else:
                                        key = tuple([int(n) for n in nums])
                            
                            if not key: continue
                            
                            if current_type not in race_payouts: race_payouts[current_type] = {}
                            race_payouts[current_type][key] = payout
                        
                    except Exception as e:
                        print(f"Parse error {current_type} row={row_list}: {e}")
                        pass
                        
        return race_payouts

    except Exception as e:
        print(f"Error scraping payout {race_id}: {e}")
        return None

def main():
    print(f"Start verifying payouts for {TARGET_DATE}...")
    
    # 1. レースID取得
    race_ids = get_race_ids(TARGET_DATE)
    if not race_ids:
        print("No races found.")
        return

    # 2. スクレイピング & 検証出力
    payouts_all = {}
    
    print(f"Processing {len(race_ids)} races...")
    
    for rid in race_ids:
        data = scrape_payout_data_from_html(rid)
        if data:
            payouts_all[rid] = data
            # 検証用に詳細出力
            print(f"--- Race {rid} ---")
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print(f"Failed {rid}")
            
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(payouts_all, f)
        
    print(f"Saved payouts to {OUTPUT_FILE} ({len(payouts_all)} races)")

if __name__ == "__main__":
    main()
