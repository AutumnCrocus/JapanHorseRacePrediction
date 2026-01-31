
import requests
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def check(rid):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
    print(f"Checking {url}...")
    
    try:
        res = requests.get(url, headers=HEADERS)
        res.encoding = 'EUC-JP'
        print(f"Status: {res.status_code}, Length: {len(res.text)}")
        
        with open(f"debug_shutuba_{rid}.html", "w", encoding="utf-8") as f:
            f.write(res.text)
        
        if "出馬表" in res.text:
            print("Page contains '出馬表'")
        else:
            print("Page does NOT contain '出馬表'")
            
        # Check for date strings
        # Look for typical date patterns
        dates = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日', res.text)
        print(f"Dates found: {dates}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with a likely ID from last week (Jan 25)
    # 2026 + 06(Nakayama) + 01(1st) + 09(9th day) + 01(1R)
    rid = "202606010901" 
    check(rid)
    
    # Also check if target date "20260131" pages exist (brute force a few likely ones)
    # Tokyo(05) 1st kai ? day ?
    # Try 1st kai, 1st day: 202605010101
    check("202605010101")
