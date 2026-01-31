
import requests
from bs4 import BeautifulSoup
import re
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def check(date_str):
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    print(f"Checking {url}...")
    
    try:
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-JP'
        print(f"Status: {res.status_code}, Length: {len(res.text)}")
        
        with open(f"debug_race_list_{date_str}.html", "w", encoding="utf-8") as f:
            f.write(res.text)
            
        soup = BeautifulSoup(res.text, 'lxml')
        links = soup.find_all('a', href=True)
        
        race_ids = []
        with open(f"debug_links_{date_str}.txt", "w", encoding="utf-8") as f:
            for link in links:
                href = link['href']
                f.write(f"{href}\n")
                if "race_id=" in href:
                    match = re.search(r'race_id=(\d+)', href)
                    if match:
                        race_ids.append(match.group(1))
                        
        print(f"Found {len(race_ids)} race IDs.")
        print(f"Sample IDs: {race_ids[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with a past date that definitely has races
    print("--- Test 20260125 (Sunday) ---")
    check("20260125")
    
    print("\n--- Test 20260131 (Target) ---")
    check("20260131")
