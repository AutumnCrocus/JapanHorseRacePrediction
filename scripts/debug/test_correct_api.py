import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
正しいパラメータでAPIをテスト
"""
import requests
import json

race_id = '202610010111'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# 正しいパラメータでAPIを呼び出し
url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&compress=0&output=json"
print(f"Testing with correct parameters: {url}\n")

response = requests.get(url, headers=HEADERS, timeout=15)
response.encoding = "EUC-JP"
data = response.json()

print(f"Status: {data.get('status')}")
print(f"Reason: {data.get('reason')}")
print(f"Update count: {data.get('update_count')}")

if 'data' in data and data['data']:
    print(f"\nData field exists!")
    data_content = data['data']
    if isinstance(data_content, dict):
        if 'odds' in data_content:
            odds = data_content.get('odds', {})
            tan_odds = odds.get('1', {})
            fuku_odds = odds.get('2', {})
            print(f"\nTan odds count: {len(tan_odds)}")
            print(f"Fuku odds count: {len(fuku_odds)}")
            
            print("\n=== Sample Tan Odds ===")
            for uma, info in list(tan_odds.items())[:3]:
                print(f"  馬{uma}: {info}")
            
            print("\n=== Sample Fuku Odds ===")
            for uma, info in list(fuku_odds.items())[:3]:
                print(f"  馬{uma}: {info}")
else:
    print("\nNo data field or data is empty")
    print(f"Full response: {json.dumps(data, ensure_ascii=False, indent=2)}")
