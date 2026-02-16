
"""
2025年東京第1回開催 限定配当データ収集スクリプト
- 目的: 2025年東京第1回開催 (ID: 20250501...) の配当データを取得し、検証を行う。
- User-Agent: 必須
- 出力: data/raw/payouts_2025.pkl (既存データにマージ)
"""

import os
import sys
import time
import pickle
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE

# ============= 設定 =============
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "payouts_2025.pkl")
INTERVAL = 1.0  # リクエスト間隔 (秒)
TARGET_PREFIX = "20250501" # 東京(05) 第1回(01)
# ===============================

def get_race_ids_tokyo01():
    """2025年東京第1回開催のレースIDリストを取得"""
    print("Loading results data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return []
        
    with open(results_path, 'rb') as f:
        df = pickle.load(f)
        
    # インデックス処理 (predict_2025_full.pyと同様)
    if isinstance(df.index, pd.Index) and df.index.name == 'race_id':
        df = df.reset_index()
    elif 'race_id' not in df.columns:
        df['race_id'] = df.index.astype(str)
    
    # race_idを文字列化
    df['race_id'] = df['race_id'].astype(str)
        
    # 東京第1回開催のみ抽出
    race_ids = df[df['race_id'].str.startswith(TARGET_PREFIX)]['race_id'].unique().tolist()
    race_ids.sort()
    
    print(f"Found {len(race_ids)} races for Tokyo 1st Holding ({TARGET_PREFIX}...).")
    return race_ids

def scrape_race_payout_direct(race_id):
    """
    netkeibaのレース結果ページから配当情報を直接取得する
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'EUC-JP'
        
        if response.status_code != 200:
            print(f"Failed to fetch {race_id}: Status {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        payouts = {}
        
        tables = soup.find_all('table')
        
        for tbl in tables:
            headers = tbl.find_all('th')
            header_texts = [th.get_text(strip=True) for th in headers]
            
            # テーブルヘッダーに配当キーワードが含まれているかチェック (単勝以外も許容)
            keywords = ['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '3連複', '三連複', '3連単', '三連単']
            if not any(k in h for h in header_texts for k in keywords):
                continue
            
            rows = tbl.find_all('tr')
            for row in rows:
                th = row.find('th')
                if not th: continue
                
                type_name = th.get_text(strip=True)
                
                key = None
                if '単勝' in type_name: key = 'tan'
                elif '複勝' in type_name: key = 'fuku'
                elif '枠連' in type_name: key = 'wakuren'
                elif '馬連' in type_name: key = 'umaren'
                elif 'ワイド' in type_name: key = 'wide'
                elif '馬単' in type_name: key = 'umatan'
                elif '3連複' in type_name or '三連複' in type_name: key = 'sanrenpuku'
                elif '3連単' in type_name or '三連単' in type_name: key = 'sanrentan'
                
                if not key: continue
                
                tds = row.find_all('td')
                if len(tds) < 2: continue
                
                horse_nums_html = tds[0]
                payouts_html = tds[1]
                
                def extract_vals(elem):
                    vals = []
                    if elem.find('ul'):
                        for li in elem.find_all('li'):
                            txt = li.get_text(strip=True).replace(',', '')
                            if txt: vals.append(txt)
                    else:
                        for txt in elem.stripped_strings:
                            t = txt.replace(',', '')
                            if t: vals.append(t)
                    if not vals:
                         txt_all = elem.get_text(' ', strip=True).replace(',', '')
                         vals = txt_all.split(' ')
                    return vals

                raw_nums = extract_vals(horse_nums_html)
                raw_pays = extract_vals(payouts_html)
                
                raw_nums = [x for x in raw_nums if x]
                raw_pays = [x for x in raw_pays if x]
                
                if key not in payouts: payouts[key] = {}
                
                if key in ['tan', 'fuku']:
                    for i in range(min(len(raw_nums), len(raw_pays))):
                        h_str = raw_nums[i]
                        if not raw_pays[i].replace('円','').isdigit(): continue
                        p_val = int(raw_pays[i].replace('円',''))
                        if p_val == 0: continue
                        try:
                            k = int(h_str)
                            payouts[key][k] = p_val
                        except: pass
                else:
                    # 複合馬券パース修正版
                    if len(raw_pays) == 1 and len(raw_nums) >= 2:
                         p_str = raw_pays[0].replace('円','')
                         if p_str.isdigit():
                             p_val = int(p_str)
                             if not any(d in ''.join(raw_nums) for d in ['-', '→', '>']):
                                 parts_int = []
                                 for x in raw_nums:
                                     if x.strip().isdigit():
                                         parts_int.append(int(x))
                                 if parts_int:
                                     if key in ['umaren', 'wide', 'wakuren', 'sanrenpuku']:
                                         k = tuple(sorted(parts_int))
                                     else:
                                         k = tuple(parts_int)
                                     payouts[key][k] = p_val
                                     continue

                    min_len = min(len(raw_nums), len(raw_pays))
                    for i in range(min_len):
                        h_str = raw_nums[i]
                        if not raw_pays[i].replace('円','').isdigit(): continue
                        p_val = int(raw_pays[i].replace('円',''))
                        
                        delimiters = ['-', '→', '>', ' ']
                        parts = [h_str]
                        for d in delimiters:
                            if d in h_str:
                                parts = h_str.split(d)
                                break
                        
                        try:
                            parts_int = tuple(int(x) for x in parts if x.strip().isdigit())
                            if not parts_int: continue
                            
                            if key in ['umaren', 'wide', 'wakuren', 'sanrenpuku']:
                                k = tuple(sorted(parts_int))
                            else:
                                k = tuple(parts_int)
                            payouts[key][k] = p_val
                        except: pass
                        
        return payouts

    except Exception as e:
        print(f"Request failed for {race_id}: {e}")
        return None

def main():
    race_ids = get_race_ids_tokyo01()
    if not race_ids:
        print("No race ids found.")
        return

    print(f"Start scraping Payouts for {len(race_ids)} races (Tokyo 1st Holding)...")
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'rb') as f:
                all_payouts = pickle.load(f)
            print(f"Loaded {len(all_payouts)} existing records.")
        except:
             all_payouts = {}
    else:
        all_payouts = {}
        
    # targets = [rid for rid in race_ids if rid not in all_payouts]
    targets = race_ids # Force re-scrape for fix
    print(f"Remaining targets: {len(targets)}")
    
    if not targets:
        print("All Tokyo targets already scraped.")
        return

    for i, race_id in enumerate(tqdm(targets)):
        data = scrape_race_payout_direct(race_id)
        if data:
            all_payouts[str(race_id)] = data
            
        if (i + 1) % 5 == 0 or (i + 1) == len(targets):
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(all_payouts, f)
            print(f"Saved {len(all_payouts)} records.")
                
        time.sleep(INTERVAL)
        
    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
