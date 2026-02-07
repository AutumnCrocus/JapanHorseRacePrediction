
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
REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/screenshots_v12"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_v12.log"),
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

def parse_marks(report_content, bets_by_race):
    """Generate marks from bets data - first horse in first bet = Honmei, etc."""
    marks_by_race = {}
    
    for race_id, bets in bets_by_race.items():
        marks = {'honmei': None, 'taiko': [], 'tanana': [], 'renka': []}
        
        # Collect all unique horses from bets
        all_horses = []
        for bet in bets:
            for h in bet['horses']:
                h_clean = str(int(h))
                if h_clean not in all_horses:
                    all_horses.append(h_clean)
        
        # Assign marks based on position
        if len(all_horses) >= 1:
            marks['honmei'] = all_horses[0]  # First = ◎
        if len(all_horses) >= 2:
            marks['taiko'].append(all_horses[1])  # Second = ○
        if len(all_horses) >= 3:
            marks['tanana'].append(all_horses[2])  # Third = ▲
        if len(all_horses) >= 4:
            marks['renka'] = all_horses[3:min(6, len(all_horses))]  # Rest = △
        
        marks_by_race[race_id] = marks
        logger.info(f"[{race_id}] Generated marks: ◎={marks['honmei']}, ○={marks['taiko']}, ▲={marks['tanana']}, △={marks['renka']}")
    
    return marks_by_race

def handle_popups(driver):
    """Handles alerts and known overlays."""
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
        logger.info("Accepted native alert.")
    except TimeoutException: pass
    
    # Handle "新規で買い目を作成しますか？" confirmation dialog (yellow/gray buttons)
    try:
        yes_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'はい')]")
        for btn in yes_btns:
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked 'はい' confirmation button.")
                time.sleep(0.5)
                return  # Exit after handling
    except: pass
    
    # Try finding and clicking default confirm button (SweetAlert)
    try:
        confirm_btns = driver.find_elements(By.CSS_SELECTOR, ".swal-button--confirm")
        for btn in confirm_btns:
            if btn.is_displayed():
                btn.click()
                logger.info("Clicked SweetAlert Confirm button.")
                time.sleep(0.5)
    except: pass
    
    # Handle jConfirm OK button
    try:
        jconfirm_btns = driver.find_elements(By.CSS_SELECTOR, ".jconfirm-buttons button")
        for btn in jconfirm_btns:
            if btn.is_displayed() and ("OK" in btn.text or "確認" in btn.text):
                btn.click()
                logger.info("Clicked jConfirm button.")
                time.sleep(0.5)
    except: pass

def diagnose_interception(driver, element):
    """Logs the element that is obscuring the target element."""
    try:
        # JS to identify element at point using Viewport Coordinates (getBoundingClientRect)
        obscurer = driver.execute_script("""
            var btn = arguments[0];
            var rect = btn.getBoundingClientRect();
            var x = rect.left + rect.width / 2;
            var y = rect.top + rect.height / 2;
            
            var el = document.elementFromPoint(x, y);
            if (el) {
                // If it's the button itself or child, that's fine? No, Intercepted means it's NOT the button.
                var info = el.tagName + '.' + el.className + ' (id=' + el.id + ')';
                try { info += ' Text:' + el.innerText.substring(0, 30); } catch(e){}
                return info;
            }
            return 'None';
        """, element)
        
        logger.error(f"⚠️ Element Intercepted by: {obscurer}")
        
        # Remove it!
        driver.execute_script("""
            var btn = arguments[0];
            var rect = btn.getBoundingClientRect();
            var x = rect.left + rect.width / 2;
            var y = rect.top + rect.height / 2;
            
            var el = document.elementFromPoint(x, y);
            if (el && el !== btn && !btn.contains(el)) { // Don't remove button or its children
                console.log('Removing obscuring element: ' + el);
                el.remove();
            }
        """, element)
        logger.info("Executed removal of obscuring element.")
        
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")

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

def set_prediction_marks(driver, race_id, marks):
    logger.info(f"[{race_id}] Setting marks...")
    def click_mark(horse_num, mark_type):
        if not horse_num: return
        try:
            row_xpath = f"//tr[td[normalize-space(text())='{horse_num}']]"
            label_xpath = f"{row_xpath}//td[contains(@class, 'Vote')]//li[contains(@class, '{mark_type}')]//label"
            elem = driver.find_element(By.XPATH, label_xpath)
            if "Selected" not in elem.get_attribute("class"):
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                time.sleep(0.1)
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
    """
    俺プロ固有のフロー:
    1. shutuba.html で「買い目を入力する」ボタンをクリック → ipat_sp.html に遷移
    2. ipat_sp.html で「買い目をセットする」ボタンをクリック
    3. 確認ダイアログで「はい」をクリック
    """
    wait = WebDriverWait(driver, 15)
    
    # Step 1: Click "買い目を入力する" button on shutuba.html
    try:
        handle_popups(driver)
        
        # Find and click "買い目を入力する" button
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
                    logger.info(f"[{race_id}] Found '買い目を入力する' button via {sel_type}={sel_value}")
                    break
                input_btn = None
            except: continue
        
        if not input_btn:
            logger.error(f"[{race_id}] Could not find '買い目を入力する' button!")
            save_evidence(driver, race_id, "input_btn_not_found")
            return False
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_btn)
        time.sleep(0.5)
        
        try:
            input_btn.click()
            logger.info(f"[{race_id}] Clicked '買い目を入力する' button.")
        except ElementClickInterceptedException:
            handle_popups(driver)
            diagnose_interception(driver, input_btn)
            time.sleep(1)
            input_btn.click()
            logger.info(f"[{race_id}] Clicked '買い目を入力する' after popup handling.")
        
        time.sleep(3)
        handle_popups(driver)
        
        # Wait for ipat_sp.html page to load
        try:
            WebDriverWait(driver, 10).until(EC.url_contains("ipat_sp.html"))
            logger.info(f"[{race_id}] Transitioned to ipat_sp.html successfully.")
        except TimeoutException:
            # Check if already on ipat_sp.html or if popup appeared
            if "ipat_sp" in driver.current_url:
                logger.info(f"[{race_id}] Already on ipat_sp.html.")
            else:
                handle_popups(driver)
                # Try new window
                if len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    if "ipat_sp" in driver.current_url:
                        logger.info(f"[{race_id}] Switched to new window with ipat_sp.html.")
                else:
                    save_evidence(driver, race_id, "ipat_transition_fail")
                    return False
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to click '買い目を入力する': {e}")
        save_evidence(driver, race_id, "input_click_fail")
        return False
    
    # Step 1.5: Input bets on ipat_sp.html (Select type, horses, amount, add)
    try:
        handle_popups(driver)
        save_evidence(driver, race_id, "before_bet_input")
        
        type_map = {"単勝": "単勝", "複勝": "複勝", "枠連": "枠連", "馬連": "馬連", 
                    "ワイド": "ワイド", "馬単": "馬単", "3連複": "3連複", "3連単": "3連単"}
        
        for i, bet in enumerate(bets):
            try:
                handle_popups(driver)
                logger.info(f"[{race_id}] Processing bet {i+1}: {bet}")
                
                # Select Bet Type
                b_type = type_map.get(bet['type'], bet['type'])
                type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
                type_clicked = False
                for btn in type_btns:
                    if b_type in btn.text:
                        btn.click()
                        type_clicked = True
                        logger.info(f"[{race_id}] Selected bet type: {b_type}")
                        break
                if not type_clicked:
                    logger.warning(f"[{race_id}] Type button {b_type} not found, skipping bet")
                    continue
                
                time.sleep(0.5)
                
                # Select Horses
                for h in bet['horses']:
                    h_clean = str(int(h))
                    try:
                        # Try multiple selectors for horse selection
                        horse_selectors = [
                            f"//table[contains(@class, 'HorseList')]//tr[td[normalize-space(text())='{h_clean}']]",
                            f"//tr[td[normalize-space(text())='{h_clean}']]",
                            f"//td[normalize-space(text())='{h_clean}']/parent::tr",
                        ]
                        horse_row = None
                        for sel in horse_selectors:
                            try:
                                horse_row = driver.find_element(By.XPATH, sel)
                                if horse_row.is_displayed():
                                    break
                            except: continue
                        
                        if horse_row:
                            if "selected" not in (horse_row.get_attribute("class") or "").lower():
                                horse_row.click()
                                logger.info(f"[{race_id}] Selected horse: {h_clean}")
                        else:
                            logger.warning(f"[{race_id}] Could not find horse row for: {h_clean}")
                    except Exception as e:
                        logger.warning(f"[{race_id}] Error selecting horse {h_clean}: {e}")
                
                time.sleep(0.5)
                
                # Input Amount
                try:
                    money_input = driver.find_element(By.NAME, "money")
                    money_val = str(int(bet['amount']) // 100)  # Convert to 100-yen units
                    
                    money_input.clear()
                    time.sleep(0.1)
                    money_input.send_keys(money_val)
                    logger.info(f"[{race_id}] Entered amount: {money_val} (x100 yen)")
                except Exception as e:
                    logger.warning(f"[{race_id}] Could not input amount: {e}")
                
                time.sleep(0.5)
                
                # Click Add button
                try:
                    add_btn = None
                    add_selectors = [
                        (By.XPATH, "//button[contains(text(), '追加')]"),
                        (By.XPATH, "//button[text()='追加']"),
                        (By.CSS_SELECTOR, "button.Common_Btn"),
                        (By.CSS_SELECTOR, "button.Add_Btn"),
                        (By.CSS_SELECTOR, ".ipat_Bet_Input button"),
                    ]
                    for sel_type, sel_value in add_selectors:
                        try:
                            add_btn = driver.find_element(sel_type, sel_value)
                            if add_btn.is_displayed():
                                logger.info(f"[{race_id}] Found Add button via {sel_type}={sel_value}")
                                break
                            add_btn = None
                        except: continue
                    
                    if add_btn:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", add_btn)
                        time.sleep(0.3)
                        try:
                            add_btn.click()
                        except:
                            driver.execute_script("arguments[0].click();", add_btn)
                        logger.info(f"[{race_id}] Clicked 'Add' button for bet {i+1}")
                        time.sleep(1)
                        handle_popups(driver)
                    else:
                        logger.warning(f"[{race_id}] Could not find Add button!")
                        save_evidence(driver, race_id, "add_btn_not_found")
                except Exception as e:
                    logger.warning(f"[{race_id}] Could not click add button: {e}")
                    
            except Exception as e:
                logger.error(f"[{race_id}] Bet {i+1} failed: {e}")
                save_evidence(driver, race_id, f"bet_{i+1}_fail")
                handle_popups(driver)
        
        save_evidence(driver, race_id, "after_bet_input")
        logger.info(f"[{race_id}] Finished inputting all bets.")
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to input bets: {e}")
        save_evidence(driver, race_id, "bet_input_fail")
        return False
    
    # Step 2: Click "買い目をセットする" button on ipat_sp.html
    try:
        handle_popups(driver)
        save_evidence(driver, race_id, "before_set_btn")  # Debug screenshot
        
        # Wait for page to fully load
        time.sleep(2)
        
        # Find and click "買い目をセットする" button
        set_btn = None
        selectors = [
            (By.XPATH, "//button[contains(text(), '買い目をセットする')]"),
            (By.XPATH, "//a[contains(text(), '買い目をセットする')]"),
            (By.XPATH, "//*[contains(text(), '買い目をセットする')]"),
            (By.CSS_SELECTOR, ".ipat_Set_Menu button.Set_Btn"),
            (By.CSS_SELECTOR, ".ipat_Set_Menu button"),
            (By.CLASS_NAME, "Set_Btn"),
            (By.CLASS_NAME, "SetBtn"),
        ]
        
        for sel_type, sel_value in selectors:
            try:
                set_btn = driver.find_element(sel_type, sel_value)
                if set_btn.is_displayed():
                    logger.info(f"[{race_id}] Found '買い目をセットする' button via {sel_type}={sel_value}")
                    break
                set_btn = None
            except: continue
        
        if not set_btn:
            logger.error(f"[{race_id}] Could not find '買い目をセットする' button!")
            save_evidence(driver, race_id, "set_btn_not_found")
            return False
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", set_btn)
        time.sleep(0.5)
        set_btn.click()
        logger.info(f"[{race_id}] Clicked '買い目をセットする' button.")
        
        time.sleep(3)
        handle_popups(driver)  # Handle any confirmation dialog
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to set bets: {e}")
        save_evidence(driver, race_id, "set_bets_fail")
        return False
    
    # Step 3: Click "この予想で勝負！" button on shutuba.html
    try:
        save_evidence(driver, race_id, "before_final_vote")
        
        # Navigate back to shutuba page if needed
        if "shutuba.html" not in driver.current_url:
            shutuba_url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
            driver.get(shutuba_url)
            time.sleep(3)
            handle_popups(driver)
        
        # Find and click "この予想で勝負！" button
        final_btn = None
        time.sleep(2)  # Wait for page to fully render
        
        selectors = [
            (By.ID, "bet_button_add"),
            (By.XPATH, "//img[@id='bet_button_add']"),
            (By.XPATH, "//img[contains(@alt, 'この予想で勝負')]/parent::*"),
            (By.XPATH, "//img[contains(@alt, '予想で勝負')]"),
            (By.XPATH, "//*[contains(text(), 'この予想で勝負')]"),
            (By.XPATH, "//button[contains(text(), '予想で勝負')]"),
            (By.XPATH, "//a[contains(text(), '予想で勝負')]"),
            (By.XPATH, "//div[contains(text(), 'この予想で勝負')]"),
            (By.CSS_SELECTOR, ".Betting_Btn"),
            (By.CSS_SELECTOR, ".BetBtn"),
            (By.CSS_SELECTOR, "button.Bet_Btn"),
            (By.CSS_SELECTOR, "a.Bet_Btn"),
        ]
        
        for sel_type, sel_value in selectors:
            try:
                elements = driver.find_elements(sel_type, sel_value)
                for elem in elements:
                    if elem.is_displayed():
                        # Check text, alt attribute, class, or accept if found by ID
                        elem_text = elem.text or ""
                        elem_alt = elem.get_attribute("alt") or ""
                        elem_class = elem.get_attribute("class") or ""
                        if "勝負" in elem_text or "勝負" in elem_alt or "Bet" in elem_class or sel_type == By.ID:
                            final_btn = elem
                            logger.info(f"[{race_id}] Found 'この予想で勝負！' button via {sel_type}={sel_value}")
                            break
                if final_btn:
                    break
            except: continue
        
        if not final_btn:
            logger.error(f"[{race_id}] Could not find 'この予想で勝負！' button!")
            save_evidence(driver, race_id, "final_btn_not_found")
            return False
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", final_btn)
        time.sleep(0.5)
        final_btn.click()
        logger.info(f"[{race_id}] Clicked 'この予想で勝負！' button.")
        
        time.sleep(3)
        handle_popups(driver)
        
        # Verify success
        save_evidence(driver, race_id, "after_final_vote")
        logger.info(f"[{race_id}] Betting completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to complete final vote: {e}")
        save_evidence(driver, race_id, "final_vote_fail")
        return False

def verify_and_retry(driver, race_id):
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
    
    # Generate marks from bets data (first horse = honmei, etc.)
    marks_by_race = parse_marks(None, bets_by_race)
        
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        for race_id in sorted(bets_by_race.keys()):
            # Skip already-started races (test with 6R onwards, ~11:10 JST start)
            if race_id < "202605010306":
                logger.info(f"[{race_id}] Skipped (already started).")
                continue
            if not bets_by_race[race_id]: continue
            
            logger.info(f"\n--- Processing {race_id} ---")
            
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                logger.info(f"[{race_id}] Attempt {attempt+1}/{max_retries}")
                
                if not ensure_shutuba_page(driver, race_id):
                    continue
                
                if race_id in marks_by_race:
                    set_prediction_marks(driver, race_id, marks_by_race[race_id])
                
                if perform_betting(driver, race_id, bets_by_race[race_id]):
                    if verify_and_retry(driver, race_id):
                        success = True
                        break
                    else:
                        logger.warning(f"[{race_id}] Verification failed. Retrying...")
                else:
                    logger.warning(f"[{race_id}] Betting flow failed. Retrying...")
                
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
