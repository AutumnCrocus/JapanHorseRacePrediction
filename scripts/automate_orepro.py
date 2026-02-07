# 設定
import json
import re
import time
import os
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
KAISAI_DATE = "20260207"
KAISAI_IDS = {
    "東京": "2026050103",
    "京都": "2026080203",
    "小倉": "2026100105"
}

def load_netkeiba_secrets():
    try:
        with open(NETKEIBA_SECRETS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {NETKEIBA_SECRETS_FILE} not found.")
        sys.exit(1)

def parse_report():
    """レポートから買い目情報を抽出"""
    bets_by_race = {} # format: {race_id: [{'type': 'ワイド', 'method': 'BOX', 'horses': [1,2], 'amount': 100}, ...]}
    
    current_venue = None
    current_race_num = None
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    regex_venue = re.compile(r"## (.+)開催")
    regex_race = re.compile(r"### (.+)(\d{2})R")
    regex_bet = re.compile(r"- \*\*(.+) (BOX|SINGLE|流し)\*\*: ([\d\-]+)(?: BOX)? \((\d+)円\)")
    
    for line in lines:
        line = line.strip()
        
        # 会場特定
        m_venue = regex_venue.match(line)
        if m_venue:
            current_venue = m_venue.group(1)
            continue
            
        # レース特定
        m_race = regex_race.match(line)
        if m_race:
            venue_name_check = m_race.group(1) # 東京, 京都...
            if venue_name_check != current_venue:
                pass # 念のため
                
            r_num = m_race.group(2)
            race_id = f"{KAISAI_IDS[current_venue]}{r_num}"
            bets_by_race[race_id] = []
            current_race_num = race_id
            continue
            
        # 買い目抽出
        m_bet = regex_bet.match(line)
        if m_bet and current_race_num:
            b_type = m_bet.group(1) # 単勝, ワイド...
            b_method = m_bet.group(2) # BOX, SINGLE
            b_horses_str = m_bet.group(3) # 14-11-1-10 or 14
            b_amount = int(m_bet.group(4))
            
            horses = [int(h) for h in b_horses_str.split('-')]
            
            bets_by_race[current_race_num].append({
                'type': b_type,
                'method': b_method,
                'horses': horses,
                'amount': b_amount
            })
            
    return bets_by_race

def setup_driver():
    options = Options()
    # options.add_argument('--headless') # 動作確認のためヘッドレスにしない
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--excludeSwitches=enable-logging")
    # ユーザーデータディレクトリは競合回避のため使用しない
    # options.add_argument(f"user-data-dir={os.getcwd()}/selenium_profile")
    
    driver = webdriver.Chrome(options=options)
    return driver

def login_netkeiba(driver, secrets):
    """Netkeibaにログイン"""
    login_url = "https://regist.netkeiba.com/account/?pid=login"
    driver.get(login_url)
    print("Trying to login to netkeiba...")
    
    try:
        wait = WebDriverWait(driver, 10)
        
        # ID入力
        print("Looking for login_id...")
        email_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        print("Found login_id.")
        
        # PASS入力
        print("Looking for pswd...")
        pass_input = wait.until(EC.presence_of_element_located((By.NAME, "pswd")))
        print("Found pswd.")
        
        # フィールドが空の場合のみ入力
        if not email_input.get_attribute('value'):
            email_input.send_keys(secrets['email'])
        
        if not pass_input.get_attribute('value'):
            pass_input.send_keys(secrets['password'])
            
        # ログインボタン押下
        print("Looking for login button...")
        try:
            # 画像ボタン待機 (XPath)
            login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='image' and @alt='ログイン']")))
            login_btn.click()
            print("Clicked login button.")
        except Exception as e:
            print(f"Login button click failed: {e}. Trying submit.")
            pass_input.submit()
        
        # ログイン後の遷移待ち
        WebDriverWait(driver, 15).until(EC.url_changes(login_url))
        print("Login successful (or redirected).")
        
    except Exception as e:
        print(f"Auto-login skipped or failed: {e}")
        print("Aborting script because login is required.")
        sys.exit(1)

def place_bets(driver, race_id, bets):
    """各レースで投票"""
    # 予想印設定ページ（起点）
    url_shutuba = f"https://orepro.netkeiba.com/bet/shutuba.html?race_id={race_id}"
    driver.get(url_shutuba)
    
    try:
        wait = WebDriverWait(driver, 10)
        
        # 「買い目を入力する」または「買い目を変更する」ボタン (#act-ipat)
        try:
            # presenceで確認してからスクロールしてクリック
            btn_input = wait.until(EC.presence_of_element_located((By.ID, "act-ipat")))
            
            # スクロールしてクリック
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn_input)
            time.sleep(1) # スクロール後の安定待ち
            btn_input.click()
            
        except Exception as e:
            print(f"Could not find '#act-ipat' button for {race_id}. Maybe already confirmed? Error: {e}")
            return

        # モーダル対応 ("新規で買い目を作成しますか？" -> "はい")
        try:
            # 少し待ってモーダルが出現するか確認
            modal_yes_btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-orange"))
            )
            # テキスト確認
            if "はい" in modal_yes_btn.text:
                modal_yes_btn.click()
                print("Clicked 'Yes' on confirmation modal.")
        except:
            pass

        # 買い目入力ページへの遷移待ち (ipat_sp.htmlが含まれるか、主要要素が出るまで)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
        time.sleep(1) # アニメーションなどの安定待ち
        
        # 買い目入力ループ
        for bet in bets:
            # 1. 券種選択 (.shikibetsu_btn)
            b_type = bet['type']
            try:
                # XPathでテキスト一致を探す
                type_btn = driver.find_element(By.XPATH, f"//li[contains(@class, 'shikibetsu_btn') and contains(text(), '{b_type}')]")
                type_btn.click()
                time.sleep(0.5)
            except Exception as e:
                print(f"Error selecting type {b_type}: {e}")
                continue
            
            # 2. 方式選択 (通常/BOX/流し)
            method_text_map = {'SINGLE': '通常', 'BOX': 'ボックス', '流し': 'ながし'} 
            target_method = method_text_map.get(bet['method'], '通常')
            
            if b_type not in ['単勝', '複勝']:
                try:
                    method_btn = driver.find_element(By.XPATH, f"//a[contains(text(), '{target_method}')]")
                    method_btn.click()
                    time.sleep(0.3)
                except:
                    pass

            # 3. 馬番選択 (label.Check01Btn)
            for h in bet['horses']:
                try:
                    horse_btns = driver.find_elements(By.XPATH, f"//label[contains(@class, 'Check01Btn') and normalize-space(text())='{h}']")
                    if horse_btns:
                        btn = horse_btns[0]
                        # check existing class
                        if "Check01Btn_Off" in btn.get_attribute("class"):
                            btn.click()
                except Exception as e:
                    print(f"Error selecting horse {h}: {e}")

            # 4. 金額入力
            try:
                kp_input = driver.find_element(By.NAME, "money")
                kp_input.clear()
                # 単位は100円=1
                amount_val = max(1, bet['amount'] // 100)
                kp_input.send_keys(str(amount_val))
            except Exception as e:
                print(f"Error inputting money: {e}")

            # 5. セット(追加)ボタン
            try:
                add_btn = driver.find_element(By.XPATH, "//button[contains(text(), '追加')]")
                add_btn.click()
                time.sleep(0.5)
            except Exception as e:
                print(f"Error clicking add set: {e}")
            
        # 6. 買い目をセットする (俺プロ画面に戻る)
        try:
            entry_btn = driver.find_element(By.CSS_SELECTOR, "button.SetBtn")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", entry_btn)
            time.sleep(0.5)
            entry_btn.click()
        except Exception as e:
            print(f"Error clicking final entry: {e}")

        
        # ⑤出馬表ページに戻る -> 「この予想で勝負！」ボタン
        wait.until(EC.url_contains("shutuba.html"))
        time.sleep(1)
        
        bet_btn_id = f"act-bet_{race_id}"
        
        try:
            final_bet_btn = wait.until(EC.element_to_be_clickable((By.ID, bet_btn_id)))
            
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", final_bet_btn)
            time.sleep(0.5)
            
            final_bet_btn.click()
            print(f"Bet placed for race {race_id}")
            
            try:
                WebDriverWait(driver, 5).until(EC.url_contains("bet_complete.html"))
            except:
                print("Warning: Did not transition to bet_complete.html, possibly already bet or error.")
            
        except Exception as e:
            print(f"Error clicking final bet button ({bet_btn_id}): {e}")
        
    except Exception as e:
        print(f"Error in race {race_id}: {e}")

def main():
    secrets = load_netkeiba_secrets()
    bets_data = parse_report()
    
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        for race_id, bets in bets_data.items():
            print(f"Processing {race_id}...")
            place_bets(driver, race_id, bets)
            
        print("All bets processed.")
        
    finally:
        print("Script finished. Close automation window manually.")
        # driver.quit()

if __name__ == "__main__":
    main()
