
import json
import re
import time
import os
import sys
import traceback
import logging
import math
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# --- Configuration ---
REPORT_FILE = "reports/prediction_20260214_ltr.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/screenshots_v15"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_v15.log"),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def load_secrets():
    with open(NETKEIBA_SECRETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_driver():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,1024")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    return driver

def save_evidence(driver, race_id, step_name):
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        path_img = os.path.join(LOG_DIR, f"{race_id}_{timestamp}_{step_name}.png")
        driver.save_screenshot(path_img)
        logger.info(f"📸 Saved screenshot: {path_img}")
        path_dom = os.path.join(LOG_DIR, f"{race_id}_{step_name}_dom.html")
        with open(path_dom, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")

def login_netkeiba(driver, secrets):
    logger.info("[LOGIN] Acccessing login page...")
    driver.get("https://regist.netkeiba.com/account/?pid=login")
    wait = WebDriverWait(driver, 10)
    try:
        user_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        pass_input = driver.find_element(By.NAME, "pswd")
        user_input.send_keys(secrets['email'])
        pass_input.send_keys(secrets['password'])
        try:
            login_btn = driver.find_element(By.XPATH, "//input[@type='image' and contains(@alt, 'ログイン')]")
        except:
            login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
        login_btn.click()
        wait.until(EC.url_contains("netkeiba.com"))
        logger.info("[LOGIN] Success.")
    except Exception as e:
        logger.error(f"[LOGIN] Failed: {e}")
        save_evidence(driver, "login_failure", "error")
        sys.exit(1)

def parse_report():
    bets_by_race = {}
    current_race_id = None
    
    if not os.path.exists(REPORT_FILE):
        logger.error(f"Report file not found: {REPORT_FILE}")
        return {}

    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            m_race = re.search(r"### (東京|京都|小倉)(\d+)R", line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))
                # 2026-02-14 (Sat)
                if venue_name == "東京": base = "2026050105"
                elif venue_name == "京都": base = "2026080205"
                elif venue_name == "小倉": base = "2026100107"
                current_race_id = f"{base}{race_num:02d}"
                bets_by_race[current_race_id] = []
                continue
            if current_race_id:
                if "- **" in line:
                    try:
                        # Manual parsing
                        parts = line.split("**: ")
                        if len(parts) < 2: continue
                        
                        left = parts[0].replace("- **", "") 
                        right = parts[1]
                        
                        bet_type, method = left.split(" ")
                        
                        m_amt = re.search(r"\((\d+)円\)", right)
                        amount = int(m_amt.group(1)) if m_amt else 100
                        
                        horses_part = right[:m_amt.start()] if m_amt else right
                        horses_part = horses_part.replace("(BOX)", "").strip()
                        
                        if "軸:" in horses_part and "相手:" in horses_part:
                            # 流し形式
                            m_axis = re.search(r"軸:([\d,]+)", horses_part)
                            m_opp = re.search(r"相手:([\d,]+)", horses_part)
                            axis = [h.strip() for h in m_axis.group(1).split(",") if h.strip().isdigit()]
                            opponents = [h.strip() for h in m_opp.group(1).split(",") if h.strip().isdigit()]
                            horses = {"axis": axis, "opponents": opponents}
                        else:
                            # 通常/BOX形式
                            horses_part = horses_part.replace("軸:", "").replace("相手:", "").strip()
                            horses = re.split(r'[-\s,]+', horses_part)
                            horses = [h.strip() for h in horses if h.strip().isdigit()]
                        
                        bets_by_race[current_race_id].append({
                            "type": bet_type, "method": method, "horses": horses, "total_amount": amount, "raw_line": line.strip()
                        })
                    except Exception as e:
                        logger.warning(f"Parse error line '{line.strip()}': {e}")
    return bets_by_race

def calculate_combinations(bet_type, method, horses):
    if method == "BOX":
        n = len(horses)
        if bet_type == "3連複": return math.comb(n, 3)
        if bet_type == "馬連" or bet_type == "ワイド": return math.comb(n, 2)
        if bet_type == "単勝" or bet_type == "複勝": return n
    elif method == "流し":
        axis_count = len(horses.get("axis", []))
        opp_count = len(horses.get("opponents", []))
        if bet_type == "3連複":
            if axis_count == 1: return math.comb(opp_count, 2)
            if axis_count == 2: return opp_count
        if bet_type == "馬連" or bet_type == "ワイド": return opp_count
    return 1 # Fallback

def handle_popups(driver):
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
    except TimeoutException: pass
    
    try:
        yes_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'はい')]")
        for btn in yes_btns:
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
                return 
    except: pass
    
    try:
        confirm_btns = driver.find_elements(By.CSS_SELECTOR, ".swal-button--confirm")
        for btn in confirm_btns:
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
    except: pass
    
    try:
        jconfirm_btns = driver.find_elements(By.CSS_SELECTOR, ".jconfirm-buttons button")
        for btn in jconfirm_btns:
            if btn.is_displayed() and ("OK" in btn.text or "確認" in btn.text):
                btn.click()
                time.sleep(0.5)
    except: pass

def check_error_popup(driver):
    try:
        swal_text = driver.find_elements(By.CSS_SELECTOR, ".swal-text")
        for el in swal_text:
            if el.is_displayed() and el.text.strip():
                logger.error(f"Error Popup: {el.text}")
                return el.text
    except: pass
    return None

def ensure_shutuba_page(driver, race_id):
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
    logger.info(f"[{race_id}] Navigating to: {url}")
    driver.get(url)
    time.sleep(4) 
    handle_popups(driver)
    try:
        driver.find_element(By.CLASS_NAME, "Vote")
        logger.info(f"[{race_id}] Page loaded.")
        return True
    except:
        return False

def perform_betting(driver, race_id, bets):
    # --- Step 1: Input Button ---
    handle_popups(driver)
    input_btn = None
    selectors = [
        (By.XPATH, "//button[contains(text(), '買い目を入力する')]"),
        (By.XPATH, "//a[contains(text(), '買い目を入力する')]"),
        (By.ID, "act-ipat"),
    ]
    for sel_type, sel_val in selectors:
        try:
            input_btn = driver.find_element(sel_type, sel_val)
            if input_btn.is_displayed(): break
        except: continue
    
    if not input_btn:
        logger.error(f"[{race_id}] Input button not found.")
        save_evidence(driver, race_id, "input_btn_missing")
        return False
        
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_btn)
    time.sleep(0.5)
    try:
        input_btn.click()
    except ElementClickInterceptedException:
        handle_popups(driver)
        driver.execute_script("arguments[0].click();", input_btn)

    time.sleep(2)
    handle_popups(driver)
    
    # Check if we moved to IPAT page
    try:
        WebDriverWait(driver, 10).until(lambda d: "ipat_sp" in d.current_url or len(d.window_handles) > 1)
    except TimeoutException:
        logger.error(f"[{race_id}] Failed transition to IPAT.")
        return False
        
    if len(driver.window_handles) > 1: driver.switch_to.window(driver.window_handles[-1])

    # --- Step 2: Input Bets ---
    type_map = {"単勝":"単勝","複勝":"複勝","枠連":"枠連","馬連":"馬連","ワイド":"ワイド","馬単":"馬単","3連複":"3連複","3連単":"3連単"}
    
    for i, bet in enumerate(bets):
        try:
            handle_popups(driver)
            b_type = type_map.get(bet['type'], bet['type'])
            click_success = False
            for btn in driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li"):
                if b_type in btn.text:
                    btn.click()
                    click_success = True
                    break
            
            target_method = "通常"
            if bet['method'] == "BOX": target_method = "ボックス"
            elif bet['method'] == "流し": target_method = "ながし"
            
            method_area = driver.find_elements(By.XPATH, "//div[contains(text(), '方式選択')]/following-sibling::ul//li")
            for m in method_area:
                if target_method in m.text:
                    m.click()
                    time.sleep(0.5)
                    break
            
            # 馬番の選択
            if bet['method'] == "流し":
                # 軸馬の選択
                for h in bet['horses']['axis']:
                    try:
                        h_val = str(int(h))
                        # 流しの場合は軸と相手でIDが異なる uc-0-X
                        axis_id = f"uc-0-{h_val}"
                        logger.info(f"Selecting Axis: {h_val} ({axis_id})")
                        driver.execute_script(f"document.getElementById('{axis_id}').click();")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Axis {h} error: {e}")
                
                # 相手馬の選択
                for h in bet['horses']['opponents']:
                    try:
                        h_val = str(int(h))
                        # 相手は uc-1-X
                        opp_id = f"uc-1-{h_val}"
                        logger.info(f"Selecting Opponent: {h_val} ({opp_id})")
                        driver.execute_script(f"document.getElementById('{opp_id}').click();")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Opponent {h} error: {e}")
            else:
                # 通常/BOX
                for h in bet['horses']:
                    try:
                        h_val = str(int(h))
                        tr_id = f"tr_{h_val}"
                        logger.info(f"Selecting Horse: {h_val} ({tr_id})")
                        driver.execute_script(f"document.getElementById('{tr_id}').click();")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Horse {h} error: {e}")

            # 1点あたりの金額算出
            combos = calculate_combinations(bet['type'], bet['method'], bet['horses'])
            unit_price = (bet['total_amount'] // combos // 100) * 100
            if unit_price < 100: unit_price = 100
            
            logger.info(f"Combos: {combos}, Unit Price: {unit_price}")

            money_input = driver.find_element(By.NAME, "money")
            money_input.clear()
            money_input.send_keys(str(unit_price // 100))
            
            try:
                driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH, "//button[contains(text(), '追加')]"))
            except:
                try: 
                    driver.execute_script("arguments[0].click();", driver.find_element(By.CSS_SELECTOR, "button.Common_Btn"))
                except: pass
            
            time.sleep(1)
            if check_error_popup(driver): continue
        except Exception as e:
            logger.error(f"[{race_id}] Bet {i} error: {e}")

    # --- Step 3: Set Bets ---
    try:
        set_btn = None
        for s in [(By.CLASS_NAME, "SetBtn"), (By.XPATH, "//button[contains(text(), '買い目をセットする')]")]:
            try: set_btn = driver.find_element(*s); break
            except: continue
            
        if set_btn:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", set_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", set_btn)
            
            try:
                WebDriverWait(driver, 10).until(lambda d: "shutuba" in d.current_url)
                logger.info(f"[{race_id}] Returned to Shutuba.")
            except TimeoutException:
                logger.error(f"[{race_id}] Failed return to Shutuba.")
                return False
        else:
            logger.error(f"[{race_id}] SetBtn missing.")
            return False
            
    except Exception as e:
        logger.error(f"[{race_id}] Set Phase Error: {e}")
        return False

    # --- Step 4: Final Vote ---
    try:
        final_btn = None
        # act-bet_XXXXXXXXXXXX の形式
        final_btn_id = f"act-bet_{race_id}"
        try:
            final_btn = driver.find_element(By.ID, final_btn_id)
        except:
            selectors = [(By.ID, "bet_button_add"), (By.CSS_SELECTOR, ".BetBtn")]
            for s in selectors:
                 try: final_btn = driver.find_element(*s); break
                 except: continue
             
        if final_btn:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", final_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", final_btn)
            logger.info(f"[{race_id}] Final Vote Clicked.")
            time.sleep(4)
            
            err = check_error_popup(driver)
            if err:
                logger.error(f"[{race_id}] Vote Error: {err}")
                return False
            return True
        else:
            logger.error(f"[{race_id}] Final Vote missing.")
            return False
    except Exception as e:
        logger.error(f"[{race_id}] Vote Phase Error: {e}")
        return False

def main():
    secrets = load_secrets()
    bets = parse_report()
    if not bets:
        logger.error("No bets parsed from report.")
        return

    driver = setup_driver()
    try:
        login_netkeiba(driver, secrets)
        for rid in sorted(bets.keys()):
            if not bets[rid]: continue
            logger.info(f"\n--- {rid} (v15) ---")
            if ensure_shutuba_page(driver, rid):
                if perform_betting(driver, rid, bets[rid]): logger.info("SUCCESS")
                else: logger.error("FAILED")
            time.sleep(1)
    finally:
        logger.info("Automation finished. Keeping browser open for verification.")
        # driver.quit()

if __name__ == "__main__":
    main()
