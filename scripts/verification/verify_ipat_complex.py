import sys
import os
import time

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from app import load_ipat_credentials
from modules.ipat_direct_automator import IpatDirectAutomator

def test_complex_betting():
    print("=== IPAT Complex Betting Verification Start ===")
    
    # 1. Credentials
    inetid, subscriber_no, pin, pars_no = load_ipat_credentials()
    if not all([subscriber_no, pin, pars_no]):
        print("Error: Credentials not found.")
        return

    # 2. Case Definition
    # Race: 2026/2/1 Tokyo 11R
    # ID: 202605010211
    # Note: 2026 is future, but race_id parsing depends on digits.
    # 05->Tokyo, 11->11R
    race_id = "202605010211"
    
    bets = [
        # 3連単 BOX 4頭 (100円)
        {'type': '3連単', 'method': 'ボックス', 'horses': [1, 7, 9, 8], 'amount': 100},
        # 3連複 BOX 5頭 (100円)
        {'type': '3連複', 'method': 'ボックス', 'horses': [1, 7, 9, 8, 14], 'amount': 100},
        # 馬連 BOX 5頭 (100円)
        {'type': '馬連', 'method': 'ボックス', 'horses': [1, 7, 9, 8, 14], 'amount': 100},
        # 単勝 通常 1点 (600円)
        {'type': '単勝', 'method': '通常', 'horses': [1], 'amount': 600}
    ]
    
    print(f"Target Race: {race_id}")
    print(f"Bets: {len(bets)} patterns")
    
    # 3. Execution
    automator = IpatDirectAutomator(debug_mode=True) # Headless OFF (GUI Visible)
    
    try:
        # Login
        print("Logging in...")
        success, msg = automator.login(inetid, subscriber_no, pin, pars_no)
        if not success:
            print(f"Auto-Login Failed: {msg}")
            print(">>> Please LOGIN MANUALLY in the browser window.")
            print(">>> After you are logged in (Top Menu displayed), press Enter here to continue...")
            input()
            print("Continuing...")
        else:
            print("Login OK")
        
        # Vote
        print("Starting voting process (JS Optimized)...")
        start_time = time.time()
        
        success, msg = automator.vote(race_id, bets, stop_at_confirmation=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("-" * 30)
        print(f"Result: {success}")
        print(f"Message: {msg}")
        print(f"Time Taken: {elapsed:.2f} sec")
        print("-" * 30)
        
        if success:
            print("✅ Verification Successful. Please check the browser window.")
            # Keep open for a bit if user wants to see
            time.sleep(10) 
        else:
            print("❌ Verification Failed.")
            
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't close immediately so user can see
        print("Verification finished. Closing in 5 seconds...")
        time.sleep(5)
        automator.close()

if __name__ == "__main__":
    test_complex_betting()
