import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from bs4 import BeautifulSoup
from modules.constants import HEADERS

def debug_scrape():
    # 中山1R
    rid = "202606010901" 
    url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
    print(f"URL: {url}")
    
    resp = requests.get(url, headers=HEADERS)
    resp.encoding = 'EUC-JP'
    print(f"Status: {resp.status_code}")
    
    soup = BeautifulSoup(resp.text, 'lxml')
    
    # テーブルクラスを探す
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables.")
    
    for i, t in enumerate(tables):
        cls = t.get('class', [])
        print(f"Table {i} class: {cls}")
        # 行数
        rows = t.find_all('tr')
        print(f"  Rows: {len(rows)}")
        # 最初の行のテキスト
        if rows:
            print(f"  Head: {rows[0].text.strip()[:50]}")

if __name__ == '__main__':
    debug_scrape()
