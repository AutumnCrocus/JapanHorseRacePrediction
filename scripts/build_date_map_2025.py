
import requests
from bs4 import BeautifulSoup
import re
import pickle
import os
import time
import pandas as pd

OUTPUT_FILE = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\date_map_2025.pickle'

PLACE_MAP = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10'
}

def build_map():
    date_map = {} # Key: race_id_prefix (YYYYSSKKDD), Value: YYYY-MM-DD
    
    # Headers for request
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for month in range(1, 13):
        url = f"https://race.netkeiba.com/top/calendar.html?year=2025&month={month}"
        print(f"Fetching {url}...")
        try:
            resp = requests.get(url, headers=headers)
            resp.encoding = 'EUC-JP' # Netkeiba usually EUC-JP
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find calendar cells
            # Class 'RaceCell' or inside table
            # Look for links with 'kaisai_date'
            links = soup.find_all('a', href=re.compile(r'kaisai_date=2025'))
            print(f"  Found {len(links)} links.")
            
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                # print(f"    Checking: {text} | {href}")
                # Extract date from href
                # ../race/list.html?kaisai_date=20250105
                match_date = re.search(r'kaisai_date=(\d{8})', href)
                if not match_date:
                    continue
                
                ymd = match_date.group(1) # 20250105
                date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
                
                # Extract Text: "1回中山1日"
                text = link.get_text(strip=True)
                # Regex: (\d+)回(..)(\d+)日
                # Sometimes venue is 3 chars? No, usually 2.
                # Special cases?
                # "金杯" might be text? No, calendar usually shows holding info.
                
                match_info = re.search(r'(\d+)回(..)(\d+)日', text)
                if match_info:
                    kai = int(match_info.group(1))
                    place_name = match_info.group(2)
                    day = int(match_info.group(3))
                    
                    place_code = PLACE_MAP.get(place_name)
                    if not place_code:
                        # Try fuzzy match
                        for k, v in PLACE_MAP.items():
                            if k in place_name:
                                place_code = v
                                break
                    
                    if place_code:
                        key = f"2025{place_code}{kai:02d}{day:02d}"
                        date_map[key] = date_str
                    else:
                        print(f"Unknown place or map failed: '{place_name}' in '{text}'")
                else:
                    print(f"Regex failed for: '{text}' (Link: {href})")
                    
            time.sleep(1) # Polite delay
            
        except Exception as e:
            print(f"Error scraping month {month}: {e}")
            
    # Save
    print(f"Total Mapped Days: {len(date_map)}")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(date_map, f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    build_map()
