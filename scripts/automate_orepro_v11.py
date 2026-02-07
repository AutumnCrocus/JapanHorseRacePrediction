
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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, NoAlertPresentException, UnexpectedAlertPresentException

# --- Configuration ---
REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/screenshots_v11"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_v11.log"),
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
    # options.add_argument("--headless") 
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
    venue_map = {"東京": "05", "京都": "08", "小倉": "10"}
    
    regex_bet = re.compile(r"- \*\*(.+?) (BOX|SINGLE|流し)\*\*:\s?([\d\-]+)(?: BOX)? \((\d+)円\)")
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            m_race = re.search(r"## (東京|京都|小倉)(\d+)R", line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))
                if venue_name == "東京": base = "2026050103"
                elif venue_name == "京都": base = "2026080203"
                elif venue_name == "小倉": base = "2026100105"
                current_race_id = f"{base}{race_num:02d}"
                bets_by_race[current_race_id] = []
                continue
            
            if current_race_id:
                m_bet = regex_bet.search(line)
                if m_bet:
                    bets_by_race[current_race_id].append({
                        "type": m_bet.group(1),
                        "method": m_bet.group(2),
                        "horses": m_bet.group(3).split('-'),
                        "amount": int(m_bet.group(4))
                    })
    return bets_by_race

def parse_marks(report_content):
    marks_by_race = {}
    current_race_id = None
    
    lines = report_content.split('\n')
    for line in lines:
        m_race = re.search(r"## (東京|京都|小倉)(\d+)R", line)
        if m_race:
            venue_name = m_race.group(1)
            race_num = int(m_race.group(2))
            if venue_name == "東京": base = "2026050103"
            elif venue_name == "京都": base = "2026080203"
            elif venue_name == "小倉": base = "2026100105"
            current_race_id = f"{base}{race_num:02d}"
            marks_by_race[current_race_id] = {'honmei': None, 'taiko': [], 'tanana': [], 'renka': []}
            continue
            
        if current_race_id:
            if "**◎ 本命**" in line:
                m = re.search(r": (\d+)", line)
                if m: marks_by_race[current_race_id]['honmei'] = m.group(1)
            elif "**○ 対抗**" in line:
                m = re.search(r": (\d+)", line)
                if m: marks_by_race[current_race_id]['taiko'].append(m.group(1))
            elif "**▲ 単穴**" in line:
                m = re.search(r": (\d+)", line)
                if m: marks_by_race[current_race_id]['tanana'].append(m.group(1))
            elif "**△ 連下**" in line:
                m = re.search(r": (\d+)", line)
                if m: marks_by_race[current_race_id]['renka'].append(m.group(1))
    return marks_by_race

def handle_popups(driver):
    """Handles both native alerts and SweetAlert modals."""
    # 1. Native Alert
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
        logger.info("Accepted native alert.")
    except TimeoutException: pass
    
    # 2. SweetAlert (HTML Modal) - Robust Handling
    try:
        # Try finding and clicking default confirm button
        confirm_btns = driver.find_elements(By.CSS_SELECTOR, ".swal-button--confirm")
        clicked = False
        for btn in confirm_btns:
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked SweetAlert Confirm button via Element.")
                clicked = True
                time.sleep(0.5)
        
        # If still present or failed to click, use JS to remove overlay
        driver.execute_script("""
            var overlays = document.querySelectorAll('.swal-overlay--show-modal');
            if (overlays.length > 0) {
                overlays.forEach(function(el){ el.remove(); });
                return true;
            }
            return false;
        """)
        # logger.info("Executed JS to remove SweetAlert overlays.") # Reduce log noise
        
    except Exception as e:
        logger.warning(f"Error handling SweetAlert: {e}")

def ensure_shutuba_page(driver, race_id):
    """Refreshes and ensures we are on the Shutuba page."""
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

def set_prediction_marks(driver, race_id, marks):
    logger.info(f"[{race_id}] Setting marks...")
    def click_mark(horse_num, mark_type):
        if not horse_num: return
        try:
            row_xpath = f"//tr[td[normalize-space(text())='{horse_num}']]"
            label_xpath = f"{row_xpath}//td[contains(@class, 'Vote')]//li[contains(@class, '{mark_type}')]//label"
            elem = driver.find_element(By.XPATH, label_xpath)
            if "Selected" not in elem.get_attribute("class"):
                elem.click()
                time.sleep(0.1)
        except Exception as e:
            logger.warning(f"[{race_id}] Failed to mark {mark_type} for {horse_num}: {e}")
            handle_popups(driver)

    click_mark(marks['honmei'], "Honmei")
    for h in marks['taiko']: click_mark(h, "Taiko")
    for h in marks['tanana']: click_mark(h, "Tanana")
    for h in marks['renka']: click_mark(h, "Renka")

def perform_betting(driver, race_id, bets):
    """Executes the betting flow. Returns True if successful, False otherwise."""
    wait = WebDriverWait(driver, 15)
    
    # 1. Open IPAT
    try:
        # Close any lingering popups
        handle_popups(driver)
        
        # Check if IPAT button is present
        btn = wait.until(EC.presence_of_element_located((By.ID, "act-ipat")))
        
        # Scroll to center
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
        time.sleep(1)
        
        click_success = False
        # Retry Click Strategy
        for click_attempt in range(3):
            try:
                # Strategy 1: Standard Click
                try:
                    btn.click()
                    logger.info(f"[{race_id}] Clicked IPAT button (Standard).")
                except ElementClickInterceptedException as e:
                    logger.warning(f"[{race_id}] Intercepted: {e}")
                    handle_popups(driver) # Try closing popup
                    # Strategy 2: ActionChains
                    from selenium.webdriver.common.action_chains import ActionChains
                    ActionChains(driver).move_to_element(btn).click().perform()
                    logger.info(f"[{race_id}] Clicked IPAT button (ActionChains).")
                
                # Check transition
                try:
                    WebDriverWait(driver, 5).until(EC.url_contains("ipat_sp.html"))
                    click_success = True
                    break # Success
                except TimeoutException:
                    logger.warning(f"[{race_id}] Transition failed, checking windows...")
                    handle_popups(driver)
                    
                    if len(driver.window_handles) > 1:
                        driver.switch_to.window(driver.window_handles[-1])
                        if "ipat_sp" in driver.current_url:
                            click_success = True
                            break
                    
                    # Strategy 3: JS Click (Last Resort)
                    logger.warning(f"[{race_id}] Trying JS Click fallback...")
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(3)
                    if "ipat_sp" in driver.current_url: 
                        click_success = True
                        break
                        
            except Exception as click_err:
                logger.warning(f"[{race_id}] Click loop error: {click_err}")
                time.sleep(1)
        
        if not click_success:
             raise Exception("Failed to transition to IPAT page after retries.")

        # Finally Wait for Load
        wait.until(EC.url_contains("ipat_sp.html"))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.Col4 li")))
        logger.info(f"[{race_id}] IPAT Interface Loaded.")
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to open IPAT: {e}")
        save_evidence(driver, race_id, "ipat_open_fail")
        return False

    # 2. Input Bets
    type_map = {"単勝": "単勝", "複勝": "複勝", "枠連": "枠連", "馬連": "馬連", "ワイド": "ワイド", "馬単": "馬単", "3連複": "3連複", "3連単": "3連単"}
    
    for i, bet in enumerate(bets):
        try:
            handle_popups(driver)
            
            # Select Type
            b_type = type_map.get(bet['type'], bet['type'])
            type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
            clicked = False
            for btn in type_btns:
                if b_type in btn.text:
                    btn.click()
                    clicked = True
                    break
            if not clicked:
                raise Exception(f"Type button {b_type} not found")
            
            time.sleep(0.5)
            
            # Select Horses
            for h in bet['horses']:
                # Clean horse number (remove leading zeros if needed, though log suggests no leading zeros)
                h_clean = str(int(h)) 
                row_xpath = f"//table[contains(@class, 'RaceOdds_HorseList_Table')]//tr[td[normalize-space(text())='{h_clean}']]"
                row = driver.find_element(By.XPATH, row_xpath)
                if "selected" not in row.get_attribute("class"):
                    row.click()
            
            time.sleep(0.5)
            
            # Input Amount (Robust)
            money_input = driver.find_element(By.NAME, "money")
            money_val = str(int(bet['amount']) // 100)
            
            # Retry Input
            for inp_retry in range(3):
                money_input.clear()
                time.sleep(0.1)
                money_input.send_keys(money_val)
                time.sleep(0.2)
                if money_input.get_attribute("value") == money_val:
                    break
                logger.warning(f"[{race_id}] Money input mismatch. Retrying... ({inp_retry})")
            
            if money_input.get_attribute("value") != money_val:
                 # JS fallback
                 driver.execute_script("arguments[0].value = arguments[1];", money_input, money_val)
            
            # Add
            add_btn = driver.find_element(By.XPATH, "//button[contains(text(), '追加')]")
            add_btn.click()
            time.sleep(1)
            
            # Confirm Alert/Popup (Crucial for "Money/Horse missing" error)
            handle_popups(driver)
            
        except Exception as e:
            logger.error(f"[{race_id}] Bet {i+1} failed: {e}")
            save_evidence(driver, race_id, f"bet_{i+1}_fail")
            # handle potential blocking overlay
            handle_popups(driver)

    # 3. Set Bets
    try:
        handle_popups(driver)
        driver.find_element(By.CLASS_NAME, "SetBtn").click()
        time.sleep(3)
        handle_popups(driver) # Check for success/error
    except Exception as e:
        logger.error(f"[{race_id}] Failed to Set Bets: {e}")
        return False

    # 4. Final Vote
    try:
        final_btn = wait.until(EC.element_to_be_clickable((By.ID, f"act-bet_{race_id}")))
        final_btn.click()
        
        time.sleep(1)
        handle_popups(driver) # Accept 'Vote?' verification
        
        time.sleep(5)
        handle_popups(driver) # Accept 'Success' or 'Error'
        
        # Check URL or Page Source for success
        if "complete" in driver.current_url or "リクエストを受け付けました" in driver.page_source:
            logger.info(f"[{race_id}] Vote Submitted.")
            return True
        else:
            logger.warning(f"[{race_id}] Vote Submission ambiguous.")
            return True # Assume success to proceed to verification
            
    except Exception as e:
        logger.error(f"[{race_id}] Final Vote Failed: {e}")
        return False

def verify_and_retry(driver, race_id):
    """
    Reloads the page and checks if bets exist in 'ul.Kaime_List'.
    Returns True if bets exist, False otherwise.
    """
    logger.info(f"[{race_id}] Verifying bets...")
    ensure_shutuba_page(driver, race_id)
    
    try:
        kaime_list = driver.find_elements(By.CSS_SELECTOR, "ul.Kaime_List li")
        if len(kaime_list) > 0:
            logger.info(f"[{race_id}] VERIFICATION SUCCESS: {len(kaime_list)} bets found.")
            return True
        else:
            logger.warning(f"[{race_id}] VERIFICATION FAILED: No bets found.")
            return False
    except Exception as e:
        logger.error(f"[{race_id}] Verification Error: {e}")
        return False

def main():
    secrets = load_secrets()
    bets_by_race = parse_report()
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        marks_by_race = parse_marks(f.read())
        
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        for race_id in sorted(bets_by_race.keys()):
            if not bets_by_race[race_id]: continue
            
            logger.info(f"\n--- Processing {race_id} ---")
            
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                logger.info(f"[{race_id}] Attempt {attempt+1}/{max_retries}")
                
                # 1. Reset/Load Page & Set Marks
                if not ensure_shutuba_page(driver, race_id):
                    continue
                
                # Verify if already betting exists?
                # If retrying, we might want to skip if already success?
                # But verifying is safer.
                
                if race_id in marks_by_race:
                    set_prediction_marks(driver, race_id, marks_by_race[race_id])
                
                # 2. Vote
                if perform_betting(driver, race_id, bets_by_race[race_id]):
                    # 3. Verify
                    if verify_and_retry(driver, race_id):
                        success = True
                        break
                    else:
                        logger.warning(f"[{race_id}] Verification failed. Retrying...")
                else:
                    logger.warning(f"[{race_id}] Betting flow failed. Retrying...")
                
                # Small cool-down before retry
                time.sleep(5)
                
            if not success:
                logger.error(f"[{race_id}] FAILED after {max_retries} attempts.")
                save_evidence(driver, race_id, "final_fail")
            
    except Exception as e:
        logger.error(f"Main Loop Error: {traceback.format_exc()}")
    finally:
        driver.quit()
        logger.info("Script finished.")

if __name__ == "__main__":
    main()
