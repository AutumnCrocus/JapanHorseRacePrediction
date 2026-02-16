
import requests
from bs4 import BeautifulSoup
import pickle
import os

def debug_scrape(race_id):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'EUC-JP'
        print(f"Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 配当テーブルを探す
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables.")
        
        for i, tbl in enumerate(tables):
            classes = tbl.get('class', [])
            print(f"Table {i} classes: {classes}")
            
            # thをチェック
            headers = tbl.find_all('th')
            header_texts = [th.get_text(strip=True) for th in headers]
            # print(f"  Headers: {header_texts[:5]}...") 
            
            # 単勝が含まれているか
            if any('単勝' in h for h in header_texts):
                print(f"  -> Likely Payout Table! (Table {i})")
                rows = tbl.find_all('tr')
                for row_idx, row in enumerate(rows):
                    text = row.get_text(strip=True)
                    print(f"    Row {row_idx}: {text[:100]}...")

    except Exception as e:
        print(f"Error: {e}")

def get_first_race_id():
    # 簡易的にresultsファイルパスを構築して読み込む
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, '..', 'data', 'raw', 'results.pkl') # 前提パス
    
    if os.path.exists(results_path):
        import pandas as pd
        with open(results_path, 'rb') as f:
            df = pickle.load(f)
        if isinstance(df.index, pd.Index) and df.index.name == 'race_id':
             df = df.reset_index()
        df['race_id'] = df.index.astype(str) if 'race_id' not in df.columns else df['race_id'].astype(str)
        
        # 2025年のIDを探す
        df_2025 = df[df['race_id'].str.startswith('2025')]
        if not df_2025.empty:
            return df_2025['race_id'].iloc[0]
            
    return '202506010101' # Fallback

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        RACE_ID = sys.argv[1]
    else:
        RACE_ID = get_first_race_id()
    
    print(f"Target Race ID: {RACE_ID}")
    debug_scrape(RACE_ID)
