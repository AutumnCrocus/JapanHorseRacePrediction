import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
RACE_ID = "202606010111"

URLS = [
    f"https://race.netkeiba.com/odds/index.html?race_id={RACE_ID}",
    f"https://race.netkeiba.com/odds/index.html?race_id={RACE_ID}&type=b1", # b1=単勝?
    f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={RACE_ID}&type=1",
]

for i, url in enumerate(URLS):
    print(f"Checking URL {i+1}: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "EUC-JP"
        if "1.6" in response.text or "2.3" in response.text or "単勝" in response.text: # Look for some odds-like numbers or text
            print(f"  Result: Found potential data (Length: {len(response.text)})")
            # Save for inspection
            with open(f"debug_odds_{i+1}.html", "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
             print(f"  Result: No obvious data found (Length: {len(response.text)})")
    except Exception as e:
        print(f"  Error: {e}")
