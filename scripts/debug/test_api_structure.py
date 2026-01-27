import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""
APIレスポンスの形式を確認
"""
import requests
import json

race_id = '202610010111'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=all&action=init&compress=0&output=json"
response = requests.get(url, headers=HEADERS, timeout=15)
response.encoding = "EUC-JP"
data = response.json()

print("=== API Response Structure ===\n")
print(f"Status: {data.get('status')}")
print(f"Reason: {data.get('reason')}")

if 'data' in data and data['data']:
    odds = data['data'].get('odds', {})
    
    # Check structure for each bet type
    for type_id in ['1', '2', '4', '5', '6', '7', '8']:
        type_data = odds.get(type_id, {})
        print(f"\n=== Type {type_id} ===")
        print(f"Count: {len(type_data)}")
        
        if type_data:
            # Show first item structure
            first_key = list(type_data.keys())[0]
            first_value = type_data[first_key]
            print(f"Sample key: {first_key} (type: {type(first_key)})")
            print(f"Sample value: {first_value} (type: {type(first_value)})")
