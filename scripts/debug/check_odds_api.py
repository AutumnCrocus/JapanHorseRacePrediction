
import requests
import json

def check_api():
    race_id = "202606010911"
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    print(f"Fetching: {url}")
    res = requests.get(url, headers=headers)
    res.encoding = "EUC-JP"
    
    try:
        data = res.json()
        print(f"Status: {data.get('status')}")
        print(f"Reason: {data.get('reason')}")
        
        if 'data' in data:
            d = data['data']
            print(f"Data keys: {d.keys()}")
            if 'odds' in d:
                odds = d['odds']
                print(f"Odds keys: {odds.keys()}")
                if '1' in odds: # Tan
                    print(f"Tan odds sample: {list(odds['1'].items())[:3]}")
                else:
                    print("No Tan odds found.")
        else:
            print("No 'data' field found.")
            
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(res.text[:500])

if __name__ == "__main__":
    check_api()
