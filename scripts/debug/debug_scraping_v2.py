import requests
import re
import json
import sys
import os

# Ensure modules can be imported
sys.path.insert(0, os.path.abspath("."))
from modules.scraping import HEADERS

race_id = "202606010502"
html_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"

print(f"Checking HTML: {html_url}")
r_html = requests.get(html_url, headers=HEADERS)
r_html.encoding = "EUC-JP"
text = r_html.text

print("Searching for odds-1 or ninki-1 in HTML...")
found_odds = re.search(r'id="odds-1[^"]*"[^>]*>([^<]*)<', text)
found_ninki = re.search(r'id="ninki-1[^"]*"[^>]*>([^<]*)<', text)

if found_odds:
    print(f"Found Odds in HTML: {found_odds.group(0)} -> Value: {found_odds.group(1)}")
else:
    print("Odds NOT found in HTML via requests.")

if found_ninki:
    print(f"Found Ninki in HTML: {found_ninki.group(0)} -> Value: {found_ninki.group(1)}")
else:
    print("Ninki NOT found in HTML via requests.")

print("\nChecking API...")
r_api = requests.get(api_url, headers=HEADERS)
r_api.encoding = "EUC-JP"
print(f"API Response: {r_api.text}")
try:
    data = json.loads(r_api.text)
    print(f"API Data status: {data.get('status')}")
    print(f"API Data contents: {data.get('data')}")
except:
    print("API Response is not valid JSON.")
