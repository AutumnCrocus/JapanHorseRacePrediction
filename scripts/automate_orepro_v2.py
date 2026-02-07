# 設定
import json
import re
import time
import os
import sys
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

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
    bets_by_race = {}
    current_venue = None
    current_race_num = None
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    regex_venue = re.compile(r"## (.+)開催")
    regex_race = re.compile(r"### (.+)(\d{2})R")
    regex_bet = re.compile(r"- \*\*(.+) (BOX|SINGLE|流し)\*\*: ([\d\-]+)(?: BOX)? \((\d+)円\)")
    
    for line in lines:
        line = line.strip()
        m_venue = regex_venue.match(line)
        if m_venue:
            current_venue = m_venue.group(1)
            continue
        m_race = regex_race.match(line)
        if m_race:
            r_num = m_race.group(2)
            race_id = f"{KAISAI_IDS[current_venue]}{r_num}"
            bets_by_race[race_id] = []
            current_race_num = race_id
            continue
        m_bet = regex_bet.match(line)
        if m_bet and current_race_num:
            b_type = m_bet.group(1)
            b_method = m_bet.group(2)
            b_horses_str = m_bet.group(3)
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
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--excludeSwitches=enable-logging")
    # 画面サイズを大きめに確保（スクロール漏れ防止）
    options.add_argument("--window-size=1280,1024")
    driver = webdriver.Chrome(options=options)
    return driver

def safe_click(driver, selector_type, selector_value, timeout=10):
    """クリック操作を安全に行うヘルパー"""
    wait = WebDriverWait(driver, timeout)
    try:
        elem = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        time.sleep(0.5) # Scroll stability
        wait.until(EC.element_to_be_clickable((selector_type, selector_value))).click()
        return True
    except ElementClickInterceptedException:
        print(f"Click intercepted for {selector_value}, trying JS click.")
        driver.execute_script("arguments[0].click();", elem)
        return True
    except Exception as e:
        print(f"Failed to click {selector_value}: {e}")
        return False

def login_netkeiba(driver, secrets):
    login_url = "https://regist.netkeiba.com/account/?pid=login"
    driver.get(login_url)
    print("[LOGIN] Trying to login to netkeiba...")
    
    try:
        # 既にログイン済みかチェック（ヘッダーアイコン等）
        try:
            WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CLASS_NAME, "Header_User_Icon")))
            print("[LOGIN] Already logged in.")
            return
        except:
            pass

        wait = WebDriverWait(driver, 10)
        print("[LOGIN] Inputting credentials...")
        email_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        pass_input = wait.until(EC.presence_of_element_located((By.NAME, "pswd")))
        
        if not email_input.get_attribute('value'):
            email_input.send_keys(secrets['email'])
        if not pass_input.get_attribute('value'):
            pass_input.send_keys(secrets['password'])
            
        print("[LOGIN] Clicking login button...")
        safe_click(driver, By.XPATH, "//input[@type='image' and @alt='ログイン']")
        
        WebDriverWait(driver, 20).until(EC.url_changes(login_url))
        print("[LOGIN] Login successful (or redirected).")
        
    except Exception as e:
        print(f"[LOGIN] Error: {e}")
        print("Please login manually in the opened window if failed.")
        # input("Press Enter after manual login...") 

def ensure_race_page(driver, race_id):
    """レース画面にいることを保証"""
    current_url = driver.current_url
    target_url = f"https://orepro.netkeiba.com/bet/shutuba.html?race_id={race_id}"
    if race_id not in current_url or "shutuba.html" not in current_url:
        driver.get(target_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "act-ipat")))

def place_bets_for_race(driver, race_id, bets):
    print(f"[{race_id}] Starting betting process...")
    ensure_race_page(driver, race_id)
    wait = WebDriverWait(driver, 10)
    
    try:
        # 1. 投票ボタンへ移動
        # 「買い目を入力する」 or 「買い目を変更する」
        btn = wait.until(EC.presence_of_element_located((By.ID, "act-ipat")))
        # ボタンのテキストを確認
        btn_text = btn.text.strip()
        print(f"[{race_id}] Bet button text: {btn_text}")
        
        # クリックして遷移
        safe_click(driver, By.ID, "act-ipat")
        
        # 2. IPATページ遷移確認
        # モーダルが出る場合と出ない場合があるが、最終的に ipat_sp.html になるはず
        try:
            wait.until(EC.url_contains("ipat_sp.html"))
            print(f"[{race_id}] Transitioned to IPAT page.")
        except TimeoutException:
            # モーダルが出ている可能性
            try:
                yes_btn = driver.find_element(By.CSS_SELECTOR, "button.btn-orange")
                if "はい" in yes_btn.text:
                    yes_btn.click()
                    wait.until(EC.url_contains("ipat_sp.html"))
                    print(f"[{race_id}] Handled confirmation modal.")
            except:
                print(f"[{race_id}] Failed to transition to IPAT page.")
                return

        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
        time.sleep(1) # Load stability
        
        # 3. 買い目入力
        for i, bet in enumerate(bets):
            print(f"[{race_id}] Setting bet {i+1}/{len(bets)}: {bet['type']} {bet['method']} {bet['horses']}")
            
            # 券種
            try:
                type_xpath = f"//li[contains(@class, 'shikibetsu_btn') and contains(text(), '{bet['type']}')]"
                driver.find_element(By.XPATH, type_xpath).click()
                time.sleep(0.3)
            except Exception as e:
                print(f"[{race_id}] Failed to select type {bet['type']}: {e}")
                continue
            
            # 方式
            if bet['type'] not in ['単勝', '複勝']:
                method_text_map = {'SINGLE': '通常', 'BOX': 'ボックス', '流し': 'ながし'}
                target_method = method_text_map.get(bet['method'], '通常')
                try:
                    method_xpath = f"//a[contains(text(), '{target_method}')]"
                    driver.find_element(By.XPATH, method_xpath).click()
                    time.sleep(0.3)
                except:
                    pass
            
            # 馬番 (リセットはされないので、必要なものだけ選択)
            # 一旦全馬番の選択状態を確認して解除...は非効率かつ危険。
            # OreProは「追加」方式なので、選択状態はあくまで「次の追加用」。
            # しかし、前の周回の選択が残っていると混ざる可能性がある。
            # 「馬番クリア」ボタンがあれば押すべきだが、見当たらない場合は手動解除。
            # ここでは「選択状態(Check01Btn)」の馬番全てをクリックしてOFFにするのが無難。
            try:
                # 選択されている馬番を探して解除
                selected_horses = driver.find_elements(By.CSS_SELECTOR, "label.Check01Btn:not(.Check01Btn_Off)")
                for h_btn in selected_horses:
                    h_btn.click()
            except:
                pass
            
            # 指定馬番を選択
            for h in bet['horses']:
                try:
                    h_label = driver.find_element(By.XPATH, f"//label[contains(@class, 'Check01Btn') and normalize-space(text())='{h}']")
                    if "Check01Btn_Off" in h_label.get_attribute("class"):
                        h_label.click()
                except Exception as e:
                    print(f"[{race_id}] Failed to select horse {h}: {e}")
                    
            # 金額
            try:
                money_input = driver.find_element(By.NAME, "money")
                money_input.clear()
                coins = max(1, bet['amount'] // 100)
                money_input.send_keys(str(coins))
            except:
                print(f"[{race_id}] Failed to input money.")
                
            # 追加ボタン
            if not safe_click(driver, By.XPATH, "//button[contains(text(), '追加')]"):
                print(f"[{race_id}] Failed to click ADD button.")
                
            time.sleep(0.5)

        # 4. セットして戻る
        print(f"[{race_id}] Finalizing entry...")
        if not safe_click(driver, By.CSS_SELECTOR, "button.SetBtn"):
            print(f"[{race_id}] Failed to click SET button.")
            return
            
        # 5. 出馬表で投票確定
        wait.until(EC.url_contains("shutuba.html"))
        time.sleep(1)
        
        final_bet_btn_id = f"act-bet_{race_id}"
        print(f"[{race_id}] Placing final bet...")
        if safe_click(driver, By.ID, final_bet_btn_id):
            # 完了待機
            try:
                WebDriverWait(driver, 10).until(EC.url_contains("bet_complete.html"))
                print(f"[{race_id}] Bet COMPLETED successfully.")
            except:
                print(f"[{race_id}] Warning: Did not see completion page.")
        else:
            print(f"[{race_id}] Failed to click final bet button.")

    except Exception as e:
        print(f"[{race_id}] CRTICAL ERROR: {e}")
        traceback.print_exc()

def main():
    secrets = load_netkeiba_secrets()
    bets_data = parse_report()
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        race_ids = sorted(bets_data.keys())
        total = len(race_ids)
        
        for i, race_id in enumerate(race_ids):
            print(f"\n--- Processing Race {i+1}/{total}: {race_id} ---")
            place_bets_for_race(driver, race_id, bets_data[race_id])
            time.sleep(2) # Interval
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        print("Done. Window open for inspection.")
        # driver.quit()

if __name__ == "__main__":
    main()
