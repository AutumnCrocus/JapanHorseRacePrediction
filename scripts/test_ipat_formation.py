"""
IPAT Betting Test Script
Tests: Formation and Nagashi (converted to Formation)
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from modules.ipat_direct_automator import IpatDirectAutomator

def load_credentials():
    path = os.path.join(os.path.dirname(__file__), 'debug', 'ipat_secrets.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    print("=== Starting IPAT Betting Test ===")
    creds = load_credentials()
    inetid = creds.get('inetid')
    subscriber_no = creds.get('subscriber_no')
    pin = creds.get('pin')
    pars_no = creds.get('pars_no')
    
    if not all([inetid, subscriber_no, pin, pars_no]):
        print("Credentials missing in ipat_secrets.json")
        return

    automator = IpatDirectAutomator(debug_mode=True)
    
    print("Logging in...")
    try:
        success, msg = automator.login(inetid, subscriber_no, pin, pars_no)
        if not success:
            print(f"Login failed: {msg}")
            return
    except Exception as e:
        print(f"Login exception: {e}")
        return
        
    print("Login successful.")
    
    # Race ID: Set to a valid race
    race_id = "202608020411"  # User's race
    
    # --- Test: 3連複 流し -> Formation変換 ---
    # 軸: 2, 相手: 4, 1, 7, 8, 6
    # Formation変換後: [[2], [4,1,7,8,6], [4,1,7,8,6]]
    # 10 points @ 100 yen = 1000 yen
    bets_nagashi_converted = [
        {
            "type": "3連複",
            "method": "フォーメーション",
            "horses": [[2], [4, 1, 7, 8, 6], [4, 1, 7, 8, 6]],
            "amount": 100
        }
    ]
    
    bets = bets_nagashi_converted
    test_name = "3連複 流し (Formation変換)"
    
    print(f"\n=== Testing: {test_name} ===")
    print(f"Bets: {bets}")
    
    try:
        success, msg = automator.vote(race_id, bets, stop_at_confirmation=True)
        print(f"Vote Result: Success={success}, Msg={msg}")
        
    except Exception as e:
        print(f"Vote Exception: {e}")
        import traceback
        traceback.print_exc()

    input("Press Enter to close browser...")
    automator.driver.quit()

if __name__ == "__main__":
    main()
