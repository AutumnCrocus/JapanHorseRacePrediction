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
from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException, NoAlertPresentException

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
    options.add_argument("--window-size=1280,1024")
    driver = webdriver.Chrome(options=options)
    return driver

def safe_click(driver, selector_type, selector_value, timeout=10):
    wait = WebDriverWait(driver, timeout)
    try:
        elem = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        time.sleep(0.5) 
        wait.until(EC.element_to_be_clickable((selector_type, selector_value))).click()
        return True
    except Exception as e:
        try:
            elem = driver.find_element(selector_type, selector_value)
            driver.execute_script("arguments[0].click();", elem)
            return True
        except:
            return False

def login_netkeiba(driver, secrets):
    login_url = "https://regist.netkeiba.com/account/?pid=login"
    driver.get(login_url)
    print("[LOGIN] Trying to login...")
    
    try:
        try:
            WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CLASS_NAME, "Header_User_Icon")))
            print("[LOGIN] Already logged in.")
            return
        except:
            pass

        wait = WebDriverWait(driver, 10)
        email_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        pass_input = wait.until(EC.presence_of_element_located((By.NAME, "pswd")))
        
        if not email_input.get_attribute('value'):
            email_input.send_keys(secrets['email'])
        if not pass_input.get_attribute('value'):
            pass_input.send_keys(secrets['password'])
            
        safe_click(driver, By.XPATH, "//input[@type='image' and @alt='ログイン']")
        
        WebDriverWait(driver, 20).until(EC.url_changes(login_url))
        print("[LOGIN] Success.")
        
    except Exception as e:
        print(f"[LOGIN] Error: {e}")
        sys.exit(1)

def set_prediction_marks(driver, race_id, bets):
    """予想印を設定する"""
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
    driver.get(url)
    
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.ID, "act-ipat")))
    
    all_horses = []
    for bet in bets:
        all_horses.extend(bet['horses'])
    
    unique_horses = []
    for h in all_horses:
        if h not in unique_horses:
            unique_horses.append(h)
    
    if not unique_horses:
        return

    honmei = unique_horses[0]
    others = unique_horses[1:]
    
    print(f"[{race_id}] Marking: ◎={honmei}, ○={others}")
    
    try:
        # ◎
        honmei_row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{honmei}']]//td[contains(@class, 'Mark')]//label[1]"
        safe_click(driver, By.XPATH, honmei_row_xpath)
        
        # ○
        for h in others:
            row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{h}']]//td[contains(@class, 'Mark')]//label[2]"
            safe_click(driver, By.XPATH, row_xpath)
            
    except Exception as e:
        print(f"[{race_id}] Error setting marks: {e}")

    # 「買い目を入力する」ボタンへ
    go_to_betting_page(driver, race_id)

def go_to_betting_page(driver, race_id):
    """投票画面へ遷移"""
    print(f"[{race_id}] Moving to betting page...")
    wait = WebDriverWait(driver, 10)
    
    if not safe_click(driver, By.ID, "act-ipat"):
        print(f"[{race_id}] Failed to click act-ipat.")
        return

    try:
        modal_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-orange"))
        )
        if "はい" in modal_btn.text:
            modal_btn.click()
            print(f"[{race_id}] Handled confirmation modal.")
    except TimeoutException:
        pass
    except Exception as e:
        print(f"[{race_id}] Modal handling error: {e}")

    try:
        wait.until(EC.url_contains("ipat_sp.html"))
        print(f"[{race_id}] on Betting Page.")
    except Exception as e:
        print(f"[{race_id}] Failed to reach betting page: {e}")

def place_bets_logic(driver, race_id, bets):
    """IPAT画面での入力処理 (Revised)"""
    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
    except:
        return

    for i, bet in enumerate(bets):
        try:
            print(f"[{race_id}] Setting bet {i+1}...")
            # 券種
            safe_click(driver, By.XPATH, f"//li[contains(@class, 'shikibetsu_btn') and contains(text(), '{bet['type']}')]")
            
            # 方式
            if bet['type'] not in ['単勝', '複勝']:
                method_map = {'SINGLE': '通常', 'BOX': 'ボックス', '流し': 'ながし'}
                t = method_map.get(bet['method'], '通常')
                safe_click(driver, By.XPATH, f"//a[contains(text(), '{t}')]")

            # 馬番選択
            # フォームは自動リセットされるため手動解除は不要。
            # ただし前回操作の残骸がないか念のため確認し、必要なものだけClick
            for h in bet['horses']:
                try:
                    lbl = driver.find_element(By.XPATH, f"//label[contains(@class, 'Check01Btn') and normalize-space(text())='{h}']")
                    # Check01Btn_Off がある場合＝未選択なのでクリックしてONにする
                    if "Check01Btn_Off" in lbl.get_attribute("class"):
                        lbl.click()
                    # 逆に Off がない場合＝既にONなので何もしない
                except Exception as e:
                    print(f"[{race_id}] Horse select error {h}: {e}")

            # 金額 (1=100円)
            try:
                inp = driver.find_element(By.NAME, "money")
                inp.clear()
                inp.send_keys(str(max(1, bet['amount'] // 100)))
            except:
                pass

            # 追加
            add_btn = driver.find_element(By.XPATH, "//button[contains(text(), '追加')]")
            add_btn.click()
            
            # 追加後の完了待ちとエラーチェック
            time.sleep(1.0) # wait for reset
            
            try:
                # アラートが出ているかチェック
                alert = driver.switch_to.alert
                print(f"[{race_id}] Alert detected: {alert.text}")
                alert.accept()
                time.sleep(0.5)
            except NoAlertPresentException:
                pass
            
            # フォームがリセットされたか確認（例: 金額が空になっているか）
            # もし金額が残っていたら追加失敗の可能性
            try:
                val = driver.find_element(By.NAME, "money").get_attribute("value")
                if val:
                    print(f"[{race_id}] WARNING: Form did not reset. Add might have failed.")
            except:
                pass
                
        except Exception as e:
            print(f"[{race_id}] Bet loop error: {e}")

    # セットして戻る
    print(f"[{race_id}] Setting bets...")
    safe_click(driver, By.CSS_SELECTOR, "button.SetBtn")
    
    try:
        wait.until(EC.url_contains("shutuba.html"))
    except:
        print(f"[{race_id}] Failed to return to shutuba.")

def finalize_race(driver, race_id):
    """最終投票ボタン押下"""
    print(f"[{race_id}] Finalizing...")
    btn_id = f"act-bet_{race_id}"
    wait = WebDriverWait(driver, 10)
    
    try:
        wait.until(EC.presence_of_element_located((By.ID, btn_id)))
        # スクロールして視認性を確保
        elem = driver.find_element(By.ID, btn_id)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        time.sleep(1)
        
        elem.click()
        
        try:
            wait.until(EC.url_contains("bet_complete.html"))
            print(f"[{race_id}] COMPLETE.")
        except:
            print(f"[{race_id}] Warning: No completion page.")
    except Exception as e:
        print(f"[{race_id}] Finalize error: {e}")

def main():
    secrets = load_netkeiba_secrets()
    bets_data = parse_report()
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        race_ids = sorted(bets_data.keys())
        for race_id in race_ids:
            print(f"\n--- {race_id} ---")
            set_prediction_marks(driver, race_id, bets_data[race_id])
            
            if "ipat_sp.html" in driver.current_url:
                place_bets_logic(driver, race_id, bets_data[race_id])
                
                if "shutuba.html" in driver.current_url:
                    finalize_race(driver, race_id)
            else:
                print(f"[{race_id}] Not on IPAT page, skipping bets.")
                
            time.sleep(1)
            
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
