import sys
import os
import time

import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.ipat_direct_automator import IpatDirectAutomator

def main():
    print("=== IPAT Smartphone Site Automation Verification ===")
    
    # 認証情報の読み込み
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipat_secrets.json')
    inetid = ""
    subscriber = ""
    pin = ""
    pars = ""
    
    if os.path.exists(secrets_path):
        print(f"Loading credentials from {secrets_path}...")
        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = json.load(f)
                inetid = secrets.get('inetid', '')
                subscriber = secrets.get('subscriber_no', '')
                pin = secrets.get('pin', '')
                pars = secrets.get('pars_no', '')
        except Exception as e:
            print(f"Failed to load secrets: {e}")
    
    if not (subscriber and pin and pars):
        print("Please enter IPAT credentials for testing:")
        if not inetid: inetid = input("INET-ID (Optional): ")
        if not subscriber: subscriber = input("Subscriber No: ")
        if not pin: pin = input("PIN: ")
        if not pars: pars = input("P-ARS: ")
    
    automator = IpatDirectAutomator()
    
    try:
        # 1. Login Test
        print("\n[Step 1] Testing Login...")
        success, msg = automator.login(inetid, subscriber, pin, pars)
        print(f"Login Result: {success}, {msg}")
        
        if not success:
            print("Login failed. Aborting.")
            return

        # Step 2: Navigate to Vote Page (Mock/Test)
        print("\n[Step 2] Testing Vote Page Navigation...")
        # 2026/01/31 Tokyo(05) 11R (Available in source dump)
        test_race_id = "202601310511" 
        print(f"Attempting to vote for {test_race_id} (Dry Run)...")   # 確実に開催されている日時を指定しないと動かない。
        # ユーザーに確認するか、直近の土日を指定する必要があるが、
        # 自動化は難しいので、ここではログイン確認までとするか、
        # エラー覚悟で適当なIDを投げる
        
        # 例: 2026年1月31日(土) 東京(05) 1R ※適当
        
        bets = [
            {'type': '単勝', 'horses': [1], 'amount': 100, 'method': '通常'},
            {'type': '馬連', 'horses': [2, 3, 4], 'amount': 100, 'method': 'ボックス'}
        ]
        
        print(f"Bets: {bets}")
        
        # 投票実行 (Stop at confirmation)
        success, msg = automator.vote(test_race_id, bets, stop_at_confirmation=True) # Changed test_race_id to race_id in instruction, but keeping test_race_id as it's defined.
        print(f"Vote Result: {success}, {msg}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nCleaning up...")
        # ブラウザを目視したい場合は下記をコメントアウト
        # automator.close()
        print("Done. Browser window is left open for inspection.")

if __name__ == "__main__":
    main()
