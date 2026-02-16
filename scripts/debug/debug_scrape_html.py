
import requests
from bs4 import BeautifulSoup

def debug_race(race_id):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    print(f"Fetching {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.encoding = 'EUC-JP'
    soup = BeautifulSoup(response.text, 'html.parser')
    
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables.")
    
    for i, tbl in enumerate(tables):
        rows = tbl.find_all('tr')
        headers = tbl.find_all('th')
        header_texts = [th.get_text(strip=True) for th in headers]
        
        # 配当テーブル候補
        if not any('単勝' in h for h in header_texts):
            continue
            
        print(f"--- Payout Table Candidate (Index {i}) ---")
        print(f"Headers: {header_texts}")
        
        for row in rows:
            th = row.find('th')
            if th:
                txt = th.get_text(strip=True)
                print(f"Type Label: '{txt}'")
                print(f"  Codepoints: {[ord(c) for c in txt]}")
                print(f"  '3連複' in text: {'3連複' in txt}")
                print(f"  '三連複' in text: {'三連複' in txt}")

debug_race('202505010101')
