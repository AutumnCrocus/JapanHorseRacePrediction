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
TARGET_DATE = '20260221'
OUTPUT_FILE = f"data/raw/payouts_{TARGET_DATE}.pkl"
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
    
    # 探索範囲 (東京05, 阪神09, 小倉10 等)
    places = [5, 9, 10]
    kais = range(1, 4)    # 1~3回
    days = range(1, 13)   # 1~12日目
    
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
                if rid.startswith(date_str[:4]):
                    race_ids.append(rid)
        race_ids = sorted(list(set(race_ids)))
        print(f"Found {len(race_ids)} races.")
        
        if not race_ids or len(race_ids) < 36:
            print("Trying brute-force scan...")
            race_ids = scan_race_ids_brute_force(date_str)
            
        return race_ids
    except Exception as e:
        print(f"Error fetching race IDs: {e}")
        return []

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

def scrape_payout_data_from_html(race_id):
    """
    race.netkeiba.com の結果ページから配当データを取得する
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    print(f"Fetching: {url}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.content, 'lxml', from_encoding='euc-jp')
        
        try:
            dfs = pd.read_html(str(soup))
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
            print(f"No payout tables found for {race_id}")
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
                        
                        if current_type in ['fuku', 'wide'] and len(p_splits) > 1:
                            nums = re.findall(r'\d+', umaban_raw)
                            
                            if current_type == 'fuku':
                                if len(nums) == len(p_splits):
                                    for i, price_str in enumerate(p_splits):
                                        if not price_str.isdigit(): continue
                                        key = int(nums[i])
                                        price = int(price_str)
                                        if current_type not in race_payouts: race_payouts[current_type] = {}
                                        race_payouts[current_type][key] = price
                                        
                            elif current_type == 'wide':
                                if len(nums) == len(p_splits) * 2:
                                    for i, price_str in enumerate(p_splits):
                                        if not price_str.isdigit(): continue
                                        h1 = int(nums[i*2])
                                        h2 = int(nums[i*2+1])
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
                            else:
                                key = parse_horse_numbers(umaban_raw)
                                
                            if not key: continue
                            
                            if current_type not in race_payouts: race_payouts[current_type] = {}
                            race_payouts[current_type][key] = payout
                        
                    except Exception as e:
                        pass
                        
        return race_payouts
    except Exception as e:
        print(f"Error scraping payout {race_id}: {e}")
        return None

def main():
    print(f"Start scraping payouts for {TARGET_DATE}...")
    race_ids = get_race_ids(TARGET_DATE)
    if not race_ids:
        print("No races found.")
        return

    payouts_all = {}
    print("Fetching payout data...")
    
    for rid in race_ids:
        data = scrape_payout_data_from_html(rid)
        if data:
            payouts_all[rid] = data
            print(f"Scraped {rid}: Found {list(data.keys())}")
        else:
            print(f"Failed {rid}")
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(payouts_all, f)
        
    print(f"Saved payouts to {OUTPUT_FILE} ({len(payouts_all)} races)")

if __name__ == "__main__":
    main()
