
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
LOG_DIR = "scripts/debug/screenshots_v14_6"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_v14_6.log"),
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
                if "- **" in line:
                    try:
                        # Parsing logic v14.6 (Nagashi Support)
                        parts = line.split("**: ")
                        if len(parts) < 2: continue
                        
                        left = parts[0].replace("- **", "") 
                        right = parts[1]
                        
                        bet_type, method = left.split(" ")
                        
                        m_amt = re.search(r"\((\d+)円\)", right)
                        amount = int(m_amt.group(1)) if m_amt else 100
                        
                        horses_text = right[:m_amt.start()] if m_amt else right
                        
                        axis_horses = []
                        partner_horses = []
                        all_horses = []

                        if "軸:" in horses_text and "相手:" in horses_text:
                            # Nagashi format: "軸: 1 - 相手: 4,5,6,9,11"
                            # Cleaning
                            clean_text = horses_text.replace("(BOX)", "").strip()
                            # Split by ' - ' assuming this separator from previous output
                            # But verifying simple split by keywords
                            try:
                                # Extract Axis
                                m_axis = re.search(r"軸:([\d,\s]+)", clean_text)
                                if m_axis:
                                    axis_horses = [h.strip() for h in m_axis.group(1).split(",") if h.strip().isdigit()]
                                
                                # Extract Partner
                                m_partner = re.search(r"相手:([\d,\s]+)", clean_text)
                                if m_partner:
                                    partner_horses = [h.strip() for h in m_partner.group(1).split(",") if h.strip().isdigit()]
                                    
                                all_horses = axis_horses + partner_horses
                            except Exception as e:
                                logger.warning(f"Nagashi parse warning: {e}")
                                # Fallback to digits
                                all_horses = [h.strip() for h in re.split(r'[-\s,]+', clean_text) if h.strip().isdigit()]

                        else:
                            # Box or Regular
                            clean_text = horses_text.replace("(BOX)", "").replace("軸:", "").replace("相手:", "").strip()
                            all_horses = [h.strip() for h in re.split(r'[-\s,]+', clean_text) if h.strip().isdigit()]
                        
                        bets_by_race[current_race_id].append({
                            "type": bet_type, 
                            "method": method, 
                            "horses": all_horses, 
                            "axis_horses": axis_horses,
                            "partner_horses": partner_horses,
                            "amount": amount, 
                            "raw_line": line.strip()
                        })
                    except Exception as e:
                        logger.warning(f"Parse error line '{line.strip()}': {e}")
    return bets_by_race

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

def select_horses_nagashi(driver, race_id, axis_horses, partner_horses):
    # Nagashi specific logic
    # uc-0-X = Axis, uc-1-X = Partner
    
    # 1. Select Axis
    for h in axis_horses:
        try:
            h_val = str(int(h)) 
            chk_id = f"uc-0-{h_val}" # Axis column ID
            chk = driver.find_element(By.ID, chk_id)
            lbl = driver.find_element(By.CSS_SELECTOR, f"label[for='{chk_id}']")
            
            if not chk.is_selected():
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(0.1)
                logger.info(f"  Selected Axis: {h} (id={chk_id})")
        except Exception as e:
            logger.error(f"[{race_id}] Axis selection failed for {h}: {e}")
            save_evidence(driver, race_id, f"nagashi_axis_{h}_fail")

    # 2. Select Partner
    for h in partner_horses:
        try:
            h_val = str(int(h)) 
            chk_id = f"uc-1-{h_val}" # Partner column ID
            chk = driver.find_element(By.ID, chk_id)
            lbl = driver.find_element(By.CSS_SELECTOR, f"label[for='{chk_id}']")
            
            if not chk.is_selected():
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(0.1)
                logger.info(f"  Selected Partner: {h} (id={chk_id})")
        except Exception as e:
            logger.error(f"[{race_id}] Partner selection failed for {h}: {e}")
            save_evidence(driver, race_id, f"nagashi_partner_{h}_fail")

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
    if not WebDriverWait(driver, 10).until(lambda d: "ipat_sp" in d.current_url or len(d.window_handles) > 1):
        logger.error(f"[{race_id}] Failed transition to IPAT.")
        return False
    if len(driver.window_handles) > 1: driver.switch_to.window(driver.window_handles[-1])

    logger.info(f"[{race_id}] On IPAT Page.")
    
    # Wait for Riot.js
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.Col4 li")))
    except:
        logger.warning("Riot render wait timeout, proceeding anyway...")

    # --- Step 2: Input Bets ---
    type_map = {"単勝":"単勝","複勝":"複勝","枠連":"枠連","馬連":"馬連","ワイド":"ワイド","馬単":"馬単","3連複":"3連複","3連単":"3連単"}
    
    for i, bet in enumerate(bets):
        try:
            handle_popups(driver)
            b_type = type_map.get(bet['type'], bet['type'])
            
            # Select Type
            type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
            for btn in type_btns:
                if b_type in btn.text:
                    btn.click()
                    break
            time.sleep(1)

            # Select Method
            target = "通常"
            if bet['method'] == "BOX": target = "ボックス"
            elif bet['method'] == "流し": target = "ながし"
            
            # Wait/Find method buttons
            method_area = driver.find_elements(By.XPATH, "//div[contains(text(), '方式選択')]/following-sibling::ul//li")
            method_clicked = False
            for m in method_area:
                if target in m.text:
                    m.click()
                    method_clicked = True
                    time.sleep(1.0) # Wait for UI update (Nagashi columns)
                    break
            
            if not method_clicked and target != "通常":
                 logger.warning(f"[{race_id}] Method '{target}' button not found or clicked.")

            # Select Horses
            if bet['method'] == "流し" and bet['axis_horses'] and bet['partner_horses']:
                logger.info(f"Processing Nagashi Bet: Axis={bet['axis_horses']}, Partner={bet['partner_horses']}")
                select_horses_nagashi(driver, race_id, bet['axis_horses'], bet['partner_horses'])
            else:
                # Normal / Box selection logic
                clean_target = "通常" if bet['method']=="BOX" else "通常" # Box uses same checkbox ID as Normal usually? -> Actually Box might assume normal selection. 
                # Box logic v14 used '通常' tab? No, it clicked 'ボックス'. 
                # If 'ボックス' is clicked, the UI is likely similar to Normal (mark horses to box).
                # Checking v14 logic: it just clicked 'ボックス' then clicked 'tr > label'.
                # IDs for Box/Normal are `tr_{id}` -> `input`/`label`.
                
                for h in bet['horses']:
                     try:
                         h_val = str(int(h)) 
                         tr_id = f"tr_{h_val}"
                         
                         # Note: If Box or Normal, we use tr_ID.
                         # If Nagashi logic was triggered but fell through here, it would fail.
                         
                         tr = driver.find_element(By.ID, tr_id)
                         lbl = tr.find_element(By.TAG_NAME, "label")
                         inp = tr.find_element(By.TAG_NAME, "input")
                         
                         if not inp.is_selected():
                             driver.execute_script("arguments[0].click();", lbl)
                             time.sleep(0.1)
                     except Exception as e:
                         logger.warning(f"[{race_id}] Horse {h} ({tr_id}) error: {e}")
                         save_evidence(driver, race_id, f"horse_{h_val}_fail")

            # Input Amount
            driver.find_element(By.NAME, "money").clear()
            driver.find_element(By.NAME, "money").send_keys(str(int(bet['amount']) // 100))
            
            # Add Button
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
            save_evidence(driver, race_id, f"bet_{i}_error")

    # --- Step 3: Set Bets ---
    try:
        try:
            total_span = driver.find_element(By.CSS_SELECTOR, ".BetPanelTotal span:last-child")
            total_txt = total_span.text
            total = int(total_txt.replace("円","").replace(",","").strip())
            if total == 0:
                logger.error(f"[{race_id}] Total 0. Abort.")
                save_evidence(driver, race_id, "total_zero")
                return False
        except: pass

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
        selectors = [(By.ID, "bet_button_add"), (By.CSS_SELECTOR, ".BetBtn")]
        for s in selectors:
             try: final_btn = driver.find_element(*s); break
             except: continue
             
        if final_btn:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", final_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", final_btn)
            logger.info(f"[{race_id}] Final Vote Clicked (JS).")
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
    driver = setup_driver()
    try:
        login_netkeiba(driver, secrets)
        for rid in sorted(bets.keys()):
            if not bets[rid]: continue
            logger.info(f"\n--- {rid} (v14.6) ---")
            if ensure_shutuba_page(driver, rid):
                if perform_betting(driver, rid, bets[rid]): logger.info("SUCCESS")
                else: logger.error("FAILED")
            time.sleep(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
