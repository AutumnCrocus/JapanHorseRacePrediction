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
        # print(f"Safe click failed for {selector_value}: {e}")
        try:
            # JS Fallback
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
        # Check if already logged in
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
    """予想印を設定する (◎, ○)"""
    # 予想印ページへ (mode=init)
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
    driver.get(url)
    
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.ID, "act-ipat")))
    
    # 馬ごとの印を決定
    # リストにある全ての馬を抽出
    all_horses = []
    for bet in bets:
        all_horses.extend(bet['horses'])
    
    # 重複排除しつつ順序保持
    unique_horses = []
    for h in all_horses:
        if h not in unique_horses:
            unique_horses.append(h)
    
    if not unique_horses:
        return

    # 1番目を◎(Double Circle), 他を○(Circle)
    honmei = unique_horses[0]
    others = unique_horses[1:]
    
    print(f"[{race_id}] Marking: ◎={honmei}, ○={others}")
    
    # 本命 ◎ (Input index 1 or class match)
    # 行を特定して、その中の特定列(Mark)の1番目のlabelをクリック
    try:
        # ◎
        honmei_row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{honmei}']]//td[contains(@class, 'Mark')]//label[1]"
        safe_click(driver, By.XPATH, honmei_row_xpath)
        
        # 相手 ○
        for h in others:
            # ○ (label index 2)
            row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{h}']]//td[contains(@class, 'Mark')]//label[2]"
            safe_click(driver, By.XPATH, row_xpath)
            
    except Exception as e:
        print(f"[{race_id}] Error setting marks: {e}")

    # 「買い目を入力する」ボタンへ
    go_to_betting_page(driver, race_id)

def go_to_betting_page(driver, race_id):
    """買い目入力画面への遷移処理"""
    print(f"[{race_id}] Moving to betting page...")
    wait = WebDriverWait(driver, 10)
    
    # 1. クリック
    if not safe_click(driver, By.ID, "act-ipat"):
        print(f"[{race_id}] Failed to click act-ipat.")
        return

    # 2. モーダル確認 ("新規で作成しますか？" -> Yes)
    try:
        # モーダルが出るまで少し待つ
        modal_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-orange"))
        )
        if "はい" in modal_btn.text:
            modal_btn.click()
            print(f"[{race_id}] Handled confirmation modal.")
    except TimeoutException:
        # モーダルが出ないならそのまま
        pass
    except Exception as e:
        print(f"[{race_id}] Modal handling error: {e}")

    # 3. ページ遷移確認 (ipat_sp.html)
    try:
        wait.until(EC.url_contains("ipat_sp.html"))
        print(f"[{race_id}] on Betting Page.")
    except Exception as e:
        print(f"[{race_id}] Failed to reach betting page: {e}")
        # リカバリ：URLがshutubaのままなら再試行すべきだが、今回はログのみ

def place_bets_logic(driver, race_id, bets):
    """IPAT画面での入力処理
    
    解析済みセレクタ:
    - 券種: a.shikibetsu_btn
    - 方式: div内のa要素 (通常/ボックス/ながし)
    - 馬番: tr#tr_{馬番} td.Horse_Select label (label.Check01Btn_Off/On)
    - 金額: input[name="money"]
    - 追加: button.Common_Btn (text='追加')
    """
    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
    except:
        print(f"[{race_id}] Shikibetsu btn not found, skipping bets.")
        return

    for bet in bets:
        try:
            print(f"[{race_id}] Processing: {bet['type']} {bet['method']} {bet['horses']} {bet['amount']}円")
            
            # 1. 券種選択 (単勝, ワイド, 3連複 等)
            bet_type_name = bet['type']
            # 3連複の表記違いに対応
            if bet_type_name == '三連複':
                bet_type_name = '3連複'
            elif bet_type_name == '三連単':
                bet_type_name = '3連単'
            
            clicked = safe_click(driver, By.XPATH, f"//a[contains(@class, 'shikibetsu_btn') and contains(text(), '{bet_type_name}')]")
            if not clicked:
                # フォールバック: li要素内のaの場合
                safe_click(driver, By.XPATH, f"//li//a[contains(@class, 'shikibetsu_btn') and contains(text(), '{bet_type_name}')]")
            time.sleep(0.3)
            
            # 2. 方式選択 (通常/ボックス/ながし) - 単勝・複勝以外
            if bet['type'] not in ['単勝', '複勝']:
                method_map = {'SINGLE': '通常', 'BOX': 'ボックス', '流し': 'ながし'}
                method_text = method_map.get(bet['method'], '通常')
                safe_click(driver, By.XPATH, f"//a[text()='{method_text}']")
                time.sleep(0.3)

            # 3. 既存選択を解除 (選択済みラベルをクリックして解除)
            try:
                selected_labels = driver.find_elements(By.CSS_SELECTOR, "label.Check01Btn_On")
                for label in selected_labels:
                    try:
                        driver.execute_script("arguments[0].click();", label)
                    except:
                        pass
                time.sleep(0.2)
            except:
                pass
            
            # 4. 馬番選択 (tr#tr_{馬番} の td.Horse_Select内のlabelをクリック)
            for h in bet['horses']:
                # BOXの場合は1列目、通常の場合も1列目でOK (三連複通常は別途対応が必要だが、現状BOXメイン)
                horse_xpaths = [
                    f"//tr[@id='tr_{h}']//td[contains(@class,'Horse_Select')]//label",
                    f"//tr[@id='tr_{h}']/td[contains(@class,'Horse_Select')][1]/label",
                    f"//tr[@id='tr_{h}']/td[4]/label",  # フォールバック: 4番目のtd
                ]
                clicked = False
                for xpath in horse_xpaths:
                    if safe_click(driver, By.XPATH, xpath):
                        print(f"  [+] Horse {h} checked")
                        clicked = True
                        break
                if not clicked:
                    print(f"  [!] Could not check horse {h}")
            
            time.sleep(0.3)

            # 5. 金額入力 (1=100円)
            try:
                inp = driver.find_element(By.NAME, "money")
                inp.clear()
                amount_unit = max(1, bet['amount'] // 100)
                inp.send_keys(str(amount_unit))
                # イベント発火で確実にUI反映
                driver.execute_script("""
                    arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                    arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
                """, inp)
            except Exception as e:
                print(f"  [!] Money input error: {e}")

            # 6. 「追加」ボタンをクリック
            add_clicked = safe_click(driver, By.XPATH, "//button[contains(@class, 'Common_Btn') and contains(text(), '追加')]")
            if not add_clicked:
                safe_click(driver, By.XPATH, "//button[contains(text(), '追加')]")
            time.sleep(0.5)
            print(f"  [OK] Bet added")
            
        except Exception as e:
            print(f"[{race_id}] Bet error: {e}")
            traceback.print_exc()

    # セットして戻る
    print(f"[{race_id}] Setting bets...")
    safe_click(driver, By.CSS_SELECTOR, "button.SetBtn")
    
    # 戻り待ち
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
        if safe_click(driver, By.ID, btn_id):
            # 完了確認
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
            # 1. 予想印設定 & 画面遷移
            set_prediction_marks(driver, race_id, bets_data[race_id])
            
            # 2. 買い目入力 (ここでURLチェックが入るので安全)
            if "ipat_sp.html" in driver.current_url:
                place_bets_logic(driver, race_id, bets_data[race_id])
                
                # 3. 最終確定
                if "shutuba.html" in driver.current_url:
                    finalize_race(driver, race_id)
            else:
                print(f"[{race_id}] Not on IPAT page, skipping bets.")
                
            time.sleep(1)
            
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
