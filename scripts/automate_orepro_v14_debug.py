
import json
import re
import time
import os
import sys
import traceback
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# --- Configuration ---
REPORT_FILE = "prediction_report_20260208_1000yen.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/screenshots_v14_debug"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_debug.log"),
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
    """Saves screenshot and DOM for debugging."""
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
    regex_bet = re.compile(r"- \*\*(.+?) (BOX|SINGLE|流し)\*\*:\s?(.+?)\s?(?:\(BOX\)\s?)?\((\d+)円\)")
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            m_race = re.search(r"## (東京|京都|小倉)(\d+)R", line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))
                if venue_name == "東京": base = "2026050104"
                elif venue_name == "京都": base = "2026080204"
                elif venue_name == "小倉": base = "2026100106"
                current_race_id = f"{base}{race_num:02d}"
                bets_by_race[current_race_id] = []
                continue
            
            if current_race_id:
                m_bet = regex_bet.search(line)
                if m_bet:
                    bet_type = m_bet.group(1).strip()
                    method = m_bet.group(2).strip()
                    horses_str = m_bet.group(3).strip()
                    amount = int(m_bet.group(4))
                    
                    horses = []
                    if method == "流し":
                        m_nagashi = re.search(r"軸:(\d+)\s*-\s*相手:([\d,]+)", horses_str)
                        if m_nagashi:
                            axis = m_nagashi.group(1)
                            partners = m_nagashi.group(2).split(',')
                            horses = [axis] + partners
                        else:
                            horses = re.split(r'[-\s,]+', horses_str)
                    else:
                        horses = re.split(r'[-\s,]+', horses_str)
                    
                    horses = [h.strip() for h in horses if h.strip()]
                    bets_by_race[current_race_id].append({
                        "type": bet_type,
                        "method": method,
                        "horses": horses,
                        "amount": amount,
                        "raw_line": line.strip()
                    })

    return bets_by_race

def handle_popups(driver):
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        txt = alert.text
        alert.accept()
        logger.info(f"Accepted native alert: {txt}")
    except TimeoutException: pass
    
    try:
        yes_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'はい')]")
        for btn in yes_btns:
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked 'はい' confirmation button.")
                time.sleep(0.5)
                return 
    except: pass
    
    try:
        confirm_btns = driver.find_elements(By.CSS_SELECTOR, ".swal-button--confirm")
        for btn in confirm_btns:
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked SweetAlert Confirm button.")
                time.sleep(0.5)
    except: pass
    
    try:
        jconfirm_btns = driver.find_elements(By.CSS_SELECTOR, ".jconfirm-buttons button")
        for btn in jconfirm_btns:
            if btn.is_displayed() and ("OK" in btn.text or "確認" in btn.text):
                btn.click()
                logger.info("Clicked jConfirm button.")
                time.sleep(0.5)
    except: pass

def check_error_popup(driver):
    try:
        swal_text = driver.find_elements(By.CSS_SELECTOR, ".swal-text")
        for el in swal_text:
            if el.is_displayed() and el.text.strip():
                logger.error(f"Error Popup Detected: {el.text}")
                return el.text
    except: pass
    return None

def ensure_shutuba_page(driver, race_id):
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
    logger.info(f"[{race_id}] Navigating to: {url}")
    driver.get(url)
    time.sleep(5) 
    handle_popups(driver)
    
    try:
        driver.find_element(By.CLASS_NAME, "Vote")
        logger.info(f"[{race_id}] Page loaded.")
        return True
    except:
        logger.error(f"[{race_id}] Failed to load Shutuba page.")
        save_evidence(driver, race_id, "load_fail")
        return False

def perform_betting(driver, race_id, bets):
    wait = WebDriverWait(driver, 15)
    
    # --- Step 1: Go to IPAT SP Page ---
    try:
        handle_popups(driver)
        
        input_btn = None
        selectors = [
            (By.XPATH, "//button[contains(text(), '買い目を入力する')]"),
            (By.XPATH, "//a[contains(text(), '買い目を入力する')]"),
            (By.XPATH, "//*[contains(text(), '買い目を入力する')]"),
            (By.ID, "act-ipat"),
        ]
        
        for sel_type, sel_value in selectors:
            try:
                input_btn = driver.find_element(sel_type, sel_value)
                if input_btn.is_displayed():
                    logger.info(f"[{race_id}] Found Input button via {sel_type}={sel_value}")
                    break
                input_btn = None
            except: continue
        
        if not input_btn:
            logger.error(f"[{race_id}] Could not find '買い目を入力する' button!")
            save_evidence(driver, race_id, "input_btn_not_found")
            return False
            
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_btn)
        time.sleep(0.5)
        input_btn.click()
        logger.info(f"[{race_id}] Clicked '買い目を入力する'.")
        
        time.sleep(3)
        handle_popups(driver)
        
        if "ipat_sp" not in driver.current_url:
             logger.warning(f"[{race_id}] URL did not change to ipat_sp. Checking windows.")
             if len(driver.window_handles) > 1:
                 driver.switch_to.window(driver.window_handles[-1])
        
        if "ipat_sp" not in driver.current_url:
             logger.error(f"[{race_id}] Failed transition. Cur URL: {driver.current_url}")
             save_evidence(driver, race_id, "ipat_transition_fail")
             return False

    except Exception as e:
        logger.error(f"[{race_id}] Transition Exception: {e}")
        return False

    # --- Step 2: Input Bets ---
    try:
        type_map = {"単勝": "単勝", "複勝": "複勝", "枠連": "枠連", "馬連": "馬連", 
                    "ワイド": "ワイド", "馬単": "馬単", "3連複": "3連複", "3連単": "3連単"}
        
        for i, bet in enumerate(bets):
             # ... (Identical Input Logic as v14 updated) ...
             # For brevity, I'll include the necessary logic but skip comments
             handle_popups(driver)
             b_type = type_map.get(bet['type'], bet['type'])
             type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
             for btn in type_btns:
                 if b_type in btn.text:
                     btn.click()
                     break
             
             target_method = "通常"
             if bet['method'] == "BOX": target_method = "ボックス"
             elif bet['method'] == "流し": target_method = "ながし"
             
             method_area = driver.find_elements(By.XPATH, "//div[contains(text(), '方式選択')]/following-sibling::ul//li")
             for m_btn in method_area:
                 if target_method in m_btn.text:
                     m_btn.click()
                     time.sleep(0.5)
                     break
             
             for h in bet['horses']:
                 h_val = str(int(h))
                 tr_elem = driver.find_element(By.ID, f"tr_{h_val}")
                 chk = tr_elem.find_element(By.TAG_NAME, "input")
                 lbl = tr_elem.find_element(By.TAG_NAME, "label")
                 if not chk.is_selected():
                     driver.execute_script("arguments[0].click();", lbl)
                     time.sleep(0.1)
             
             money_input = driver.find_element(By.NAME, "money")
             money_input.clear()
             money_input.send_keys(str(int(bet['amount']) // 100))
             
             try:
                 driver.find_element(By.XPATH, "//button[contains(text(), '追加')]").click()
             except:
                 driver.find_element(By.CSS_SELECTOR, "button.Common_Btn").click()
             
             time.sleep(1)
             if check_error_popup(driver):
                 continue
                 
        save_evidence(driver, race_id, "bets_input_done")
        
        # --- Step 3: Set Bets ---
        set_btn = None
        selectors = [
             (By.CLASS_NAME, "SetBtn"),
             (By.XPATH, "//button[contains(text(), '買い目をセットする')]")
        ]
        for s in selectors:
             try: set_btn = driver.find_element(*s); break
             except: continue
        
        if set_btn:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", set_btn)
            time.sleep(0.5)
            set_btn.click()
            logger.info(f"[{race_id}] Clicked Set Bets.")
            save_evidence(driver, race_id, "after_set_click") # DEBUG
        else:
            logger.error(f"[{race_id}] Set Bets button not found.")
            save_evidence(driver, race_id, "set_btn_not_found")
            return False

        time.sleep(3)
        handle_popups(driver)
        
    except Exception as e:
        logger.error(f"[{race_id}] Input Exception: {e}")
        save_evidence(driver, race_id, "input_exception")
        return False

    # --- Step 4: Final Vote ---
    try:
        # Debug: Check URL
        logger.info(f"[{race_id}] Post-Set URL: {driver.current_url}")
        save_evidence(driver, race_id, "pre_final_vote")
        
        if "shutuba.html" not in driver.current_url:
            logger.warning("Not on shutuba page yet...")
            time.sleep(3)
        
        final_btn = None
        selectors = [(By.ID, "bet_button_add"), (By.CSS_SELECTOR, ".BetBtn")]
        for s in selectors:
            try:
                final_btn = driver.find_element(*s)
                if final_btn.is_displayed(): break
            except: continue
        
        if final_btn:
            final_btn.click()
            logger.info(f"[{race_id}] Clicked Final Vote.")
            time.sleep(3)
            
            err = check_error_popup(driver)
            if err:
                 logger.error(f"[{race_id}] Final Vote Failed: {err}")
                 save_evidence(driver, race_id, "final_vote_fail_popup")
                 return False
            
            logger.info(f"[{race_id}] Vote Sequence Completed.")
            return True
        else:
            logger.error(f"[{race_id}] Final Vote Button not found.")
            save_evidence(driver, race_id, "final_btn_not_found")
            return False
            
    except Exception as e:
         logger.error(f"[{race_id}] Final Vote Exception: {e}")
         save_evidence(driver, race_id, "final_exception")
         return False

def main():
    secrets = load_secrets()
    bets_by_race = parse_report()
    driver = setup_driver()
    try:
        login_netkeiba(driver, secrets)
        # Limit to first 2 races for debug
        keys = sorted(bets_by_race.keys())[:2]
        for race_id in keys:
            if not bets_by_race[race_id]: continue
            logger.info(f"\n--- Processing {race_id} (DEBUG) ---")
            if ensure_shutuba_page(driver, race_id):
                 perform_betting(driver, race_id, bets_by_race[race_id])
            time.sleep(2)
    except Exception as e:
        logger.error(f"Main Loop Error: {traceback.format_exc()}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
