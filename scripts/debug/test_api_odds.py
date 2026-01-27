import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
オッズ取得をAPIベースで実装(簡易版テスト)
"""
import requests

race_id = '202610010111'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# Test TAN API
tan_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
print(f"Testing: {tan_url}")

response = requests.get(tan_url, headers=HEADERS, timeout=15)
response.encoding = "EUC-JP"
data = response.json()

print(f"\nStatus: {data.get('status')}")
if data.get('status') == 'result':
    odds = data.get('data', {}).get('odds', {}).get('1', {})
    print(f"Found {len(odds)} horses")
    for uma, info in list(odds.items())[:3]:
        print(f"  馬{uma}: {info}")
else:
    print(f"Full response: {data}")
