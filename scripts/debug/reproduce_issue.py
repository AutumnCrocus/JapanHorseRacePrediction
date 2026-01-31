
import requests
import json

def reproduce():
    url = "http://127.0.0.1:5018/api/predict_by_url"
    target_url = "https://race.netkeiba.com/race/shutuba.html?race_id=202606010911&rf=race_submenu"
    
    payload = {
        "url": target_url,
        "budget": 10000
    }
    
    print(f"Testing URL: {target_url}")
    try:
        response = requests.post(url, json=payload)
        
        print("Status Code:", response.status_code)
        try:
            data = response.json()
            if 'predictions' in data:
                print("\n--- Predictions Sample ---")
                for p in data['predictions'][:5]:
                    print(f"Horse: {p.get('horse_name')}, No: {p.get('horse_number')}, Odds: {p.get('odds')}, EV: {p.get('expected_value')}")

            if 'recommendations' in data:
                print("\n--- Recommendations Sample ---")
                for r in data['recommendations'][:3]:
                     print(f"Rec Type: {r.get('bet_type')}, Method: {r.get('method')}")
                     print(f"  Reason: {r.get('reason')}")
                     print("-" * 20)
            if 'error' in data:
                print(f"ERROR From Server: {data['error']}")
            elif 'success' in data and not data['success']:
                print(f"FAILURE: {data}")
            else:
                print("SUCCESS/Partial Success")
        except:
            print("Response Text:", response.text)
            
    except Exception as e:
        print(f"Request Error: {e}")

if __name__ == "__main__":
    reproduce()
