
import requests
from bs4 import BeautifulSoup
import re
import pickle
import pandas as pd

OUTPUT_FILE = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\date_map_2025.pickle'

PLACE_MAP = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10'
}

def reconstruct():
    # Helper to track (Kai, Day) for each venue
    # Logic: If date gap > 1 month, Kai++? No, hard to guess.
    # We really need the "1回中山" text.
    
    # Try alternate URL that lists "Kaisai" clearly
    # https://db.netkeiba.com/race/list/2025/
    # This lists by race_id or date?
    
    # Let's try grabbing the "Race List" page directly for a known date and see text?
    # No.
    
    # Let's try parsing the "formatted" text from calendar, accounting for bad separators
    # URL: https://race.netkeiba.com/top/calendar.html?year=2025&month={m}
    
    date_map = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Target: Schedule List (Sequential)
    url = "https://race.netkeiba.com/top/schedule.html?year=2025"
    print(f"Scraping Schedule: {url}")
    
    try:
        resp = requests.get(url, headers=headers)
        resp.encoding = 'EUC-JP'
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Text extraction (Entire body text to avoid HTML structure issues)
        # But we need row alignment (Date vs Venue).
        # Check rows. Usually <tr class="RaceListMonth">...
        
        # Netkeiba Schedule Page uses table.
        # Format:
        # Date | Venue 1 | Venue 2 ...
        # "1月5日(日)" | "1回中山1日" | "1回京都1日"
        
        # Find all cells with text like "1回..1日"
        # And associate with nearest preceding Date?
        
        # Let's iterate TRs.
        rows = soup.find_all('tr')
        print(f"Found {len(rows)} rows.")
        
        current_date_str = None
        
        for row in rows:
            text = row.get_text(separator=' ', strip=True)
            # Check for Date: "1月 5日" or "1/5"
            # Regex: (\d+)月(\d+)日
            date_match = re.search(r'(\d+)月\s*(\d+)日', text)
            if date_match:
                m = int(date_match.group(1))
                d = int(date_match.group(2))
                current_date_str = f"2025-{m:02d}-{d:02d}"
                # Row might ALSO contain venues?
            
            if current_date_str:
                # Look for Venues in this row
                # Regex: (\d+)回(..)(\d+)日
                # Find ALL matches in row text
                venues = re.findall(r'(\d+)回\s*(..)\s*(\d+)日', text)
                for (k, p_name, dy) in venues:
                    kai = int(k)
                    place_name = p_name
                    day = int(dy)
                    
                    place_code = PLACE_MAP.get(place_name)
                    if not place_code:
                         for k_map, v_map in PLACE_MAP.items():
                             if k_map in place_name:
                                 place_code = v_map
                                 break
                    
                    if place_code:
                        # Construct Key: 2025 SS KK DD
                        key = f"2025{place_code}{kai:02d}{day:02d}"
                        # Only 10 digits needed for race_id prefix?
                        # race_id: 2025 SS KK DD RR
                        # My key: 2025 SS KK DD (10 chars). Matches exactly.
                        date_map[key] = current_date_str
                        # print(f"Mapped {key} -> {current_date_str}")
                
    except Exception as e:
        print(f"Error scraping schedule: {e}")
            
    print(f"Mapped {len(date_map)} keys from Schedule.")
    # No loop over months needed, schedule page is usually full year or long list.
    # Actually Netkeiba schedule page might be paged?
    # "top/schedule.html" usually defaults to current month?
    # BUT "List/Schedule" might be better.
    # Let's assume it lists some.
    pass
            
    print(f"Mapped {len(date_map)} keys.")
    if len(date_map) > 0:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(date_map, f)
            
if __name__ == '__main__':
    reconstruct()
