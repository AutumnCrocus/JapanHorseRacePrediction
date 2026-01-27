import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import sys

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
RACE_ID = "202606010111"
URL = f"https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}"

try:
    response = requests.get(URL, headers=HEADERS)
    response.encoding = "EUC-JP"
    if response.status_code == 200:
        with open("debug_race.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Success")
    else:
        print(f"Failed: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
