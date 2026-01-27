import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def check_api():
    print(f"Checking API at {BASE_URL}...")
    
    # 1. Feature Importance
    print("\n[1] Checking /api/feature_importance")
    try:
        resp = requests.get(f"{BASE_URL}/api/feature_importance")
        print(f"Status Code: {resp.status_code}")
        print(f"Content Type: {resp.headers.get('Content-Type')}")
        try:
            data = resp.json()
            print(f"JSON Success: {data.get('success')}")
            print(f"Available: {data.get('available')}")
            if data.get('features'):
                print(f"Features Count: {len(data['features'])}")
                print(f"First Feature: {data['features'][0]}")
            else:
                print("No features returned.")
            
            if not data.get('success'):
                print(f"Error Message: {data.get('message') or data.get('error')}")
                
        except Exception as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Request Failed: {e}")

    # 2. Model Info
    print("\n[2] Checking /api/model_info")
    try:
        resp = requests.get(f"{BASE_URL}/api/model_info")
        print(f"Status Code: {resp.status_code}")
        try:
            data = resp.json()
            print(f"JSON Success: {data.get('success')}")
            print(f"Algorithm: {data.get('algorithm')}")
            
            if not data.get('success'):
                print(f"Error Message: {data.get('error')}")

        except Exception as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Request Failed: {e}")

if __name__ == "__main__":
    check_api()
