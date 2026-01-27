import requests
import re
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from modules.scraping import HEADERS

race_id = "202606010502"
url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
r = requests.get(url, headers=HEADERS)
r.encoding = "EUC-JP"

print("--- API Endpoints in HTML ---")
apis = set(re.findall(r'api_get_[\w\.]+', r.text))
for api in apis:
    print(api)

print("\n--- Testing likely candidates ---")
candidates = [
    f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1",
    f"https://race.netkeiba.com/api/api_get_yoso_odds.html?race_id={race_id}",
    f"https://race.netkeiba.com/api/api_get_odds.html?race_id={race_id}",
    f"https://race.netkeiba.com/api/api_get_shutuba_odds.html?race_id={race_id}"
]

for c in candidates:
    try:
        res = requests.get(c, headers=HEADERS)
        print(f"URL: {c}")
        print(f"Status: {res.status_code}")
        print(f"Content (first 100): {res.text[:100]}")
        print("-" * 20)
    except:
        print(f"Failed to fetch {c}")
