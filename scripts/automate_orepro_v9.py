
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
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, NoAlertPresentException

# --- Configuration ---
REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
KAISAI_DATE = "20260207"
LOG_DIR = "scripts/debug/screenshots_v9"

# Venue Mapping
KAISAI_IDS = {
    "東京": "05",
    "京都": "08",
    "小倉": "10"
}

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "automation_v9.log"),
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
    # options.add_argument("--headless") # Headless off for reliable IPAT
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,1024")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # ユーザーデータディレクトリを指定しない（クリーンな状態でログインするため）
    # options.add_argument(f"user-data-dir={os.getcwd()}/selenium_profile")

    driver = webdriver.Chrome(options=options)
    return driver

def save_evidence(driver, race_id, step_name):
    """Saves screenshot and DOM for debugging."""
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Screenshot
        path_img = os.path.join(LOG_DIR, f"{race_id}_{timestamp}_{step_name}.png")
        driver.save_screenshot(path_img)
        logger.info(f"📸 Saved screenshot: {path_img}")
        
        # DOM
        path_dom = os.path.join(LOG_DIR, f"{race_id}_{step_name}_dom.html")
        with open(path_dom, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logger.info(f"📄 Saved DOM: {path_dom}")
        
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")

def login_netkeiba(driver, secrets):
    """Log in to netkeiba."""
    logger.info("[LOGIN] Acccessing login page...")
    driver.get("https://regist.netkeiba.com/account/?pid=login")
    wait = WebDriverWait(driver, 10)
    
    try:
        user_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        pass_input = driver.find_element(By.NAME, "pswd")
        
        user_input.send_keys(secrets['email'])
        pass_input.send_keys(secrets['password'])
        
        # ログインボタン (画像ボタンの場合がある)
        try:
            login_btn = driver.find_element(By.XPATH, "//input[@type='image' and contains(@alt, 'ログイン')]")
        except:
            login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
            
        login_btn.click()
        
        # Wait for redirect
        wait.until(EC.url_contains("netkeiba.com"))
        logger.info("[LOGIN] Success.")
        
    except Exception as e:
        logger.error(f"[LOGIN] Failed: {e}")
        save_evidence(driver, "login_failure", "error")
        sys.exit(1)

def parse_report():
    """Parse prediction report for bets."""
    bets_by_race = {}
    current_race_id = None
    
    venue_map = {"東京": "05", "京都": "08", "小倉": "10"}
    
    # updated regex to match the report format: "- **ワイド BOX**: 14-11-1-10 BOX (600円)"
    # Captures: 1=Type, 2=Method, 3=Horses, 4=Amount
    regex_bet = re.compile(r"- \*\*(.+?) (BOX|SINGLE|流し)\*\*:\s?([\d\-]+)(?: BOX)? \((\d+)円\)")
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # Race Header: "## 東京1R (10:05) [Confidence: 85%]"
            m_race = re.search(r"## (東京|京都|小倉)(\d+)R", line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))
                venue_id = venue_map[venue_name]
                # Race ID format: 2026 + venue_id + kaisai_times + day + race_num
                # Hardcoded for 2026/2/7 based on task context
                # User provided: 202605010301 for Tokyo 1R.
                # 2026 + 05 (Tokyo) + 01 (1st Kai) + 03 (3rd Day)? No, date=20260207
                # Let's derive standard ID. 2026 + Venue + Kai + Day + R
                # Assuming Kai/Day from user request or fixed.
                # User request: "202605010301" -> 2026 05 01 03 01
                # Kai=01, Day=03. 
                # Let's use a helper to construct IDs correctly if we can match them.
                # OR, just map sequentially if we know the set.
                # Better: Extract from the known list in user request if possible. 
                # But here we construct it.
                # The prompt implies 2026050103xx for Tokyo.
                # 2026080203xx for Kyoto
                # 2026100105xx for Kokura (Wait, Kokura ID logic might differ)
                
                # Logic based on observed IDs in previous logs:
                # Tokyo: 2026050103xx
                # Kyoto: 2026080203xx
                # Kokura: 2026100105xx
                
                if venue_name == "東京":
                    base = "2026050103"
                elif venue_name == "京都":
                    base = "2026080203"
                elif venue_name == "小倉":
                    base = "2026100105"
                
                current_race_id = f"{base}{race_num:02d}"
                bets_by_race[current_race_id] = []
                continue
            
            # Bet Line
            if current_race_id:
                m_bet = regex_bet.search(line)
                if m_bet:
                    b_type = m_bet.group(1)
                    b_method = m_bet.group(2)
                    b_horses = m_bet.group(3).split('-')
                    b_amount = int(m_bet.group(4))
                    
                    # Store
                    bets_by_race[current_race_id].append({
                        "type": b_type,
                        "method": b_method,
                        "horses": b_horses,
                        "amount": b_amount
                    })
                    
            # Also parse "本命", "対抗" etc for setting marks
            # Format: "1. **◎ 本命**: 14 ファーストシーン"
            # We need to extract these to set marks on the page
            # We can store them in specific keys
            if current_race_id:
                if "**◎ 本命**" in line:
                    m = re.search(r": (\d+)", line)
                    if m: 
                        # Store as special "mark" bet or metadata
                        if "marks" not in bets_by_race[current_race_id]: matches = [] # Hacky storage
                        # Better: add to a 'meta' dict? 
                        # For simplicity, let's just parse the ranking list at the bottom of race section?
                        pass

    # Re-pass to extract marks more reliably
    # We need a separate pass or structure for marks.
    return bets_by_race

def parse_marks(report_content):
    """
    Parses the report content to extract prediction marks (Honmei, Taiko, etc.) for each race.
    Returns a dict: { race_id: { 'honmei': '14', 'taiko': ['11'], ... } }
    """
    marks_by_race = {}
    current_race_id = None
    venue_map = {"東京": "05", "京都": "08", "小倉": "10"}
    
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
            # Match: "1. **◎ 本命**: 14 ..."
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

def ensure_shutuba_page(driver, race_id):
    """出馬表ページ（予想入力画面）にいることを保証する"""
    max_retries = 2
    for attempt in range(max_retries):
        current_url = driver.current_url
        if "shutuba.html" in current_url and "mode=init" in current_url:
            try:
                # v9: Check for the 'Vote' column which contains buttons
                driver.find_element(By.CLASS_NAME, "Vote") 
                logger.info(f"[{race_id}] Verified Start List page (Vote column found).")
                return True
            except NoSuchElementException:
                logger.warning(f"[{race_id}] URL correct but 'Vote' column not found.")
        
        url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
        logger.info(f"[{race_id}] Navigating to: {url}")
        driver.get(url)
        time.sleep(2)
        
        try:
            driver.find_element(By.CLASS_NAME, "Vote")
            logger.info(f"[{race_id}] Verified Start List page after navigation.")
            save_evidence(driver, race_id, "01_shutuba_ok")
            return True
        except:
            logger.warning(f"[{race_id}] Attempt {attempt+1} failed to reach Shutuba page.")
            save_evidence(driver, race_id, f"01_shutuba_fail_{attempt+1}")
            
    logger.error(f"[{race_id}] Could not reach Shutuba page.")
    return False

def set_prediction_marks(driver, race_id, marks):
    """
    Sets prediction marks (◎, ○, etc.) on the Shutuba page.
    v9 Update: Uses correct selectors (li.Honmei label, etc.) inside tr matching horse number.
    NOTE: ◎ (Honmei) is REQUIRED for OrePro IPAT to function correctly.
    """
    logger.info(f"[{race_id}] Marking: ◎={marks['honmei']}, ○={marks['taiko']}")
    
    # Helper to click mark
    def click_mark(horse_num, mark_type):
        if not horse_num: return
        # mark_type: 'Honmei', 'Taiko', 'Tanana', 'Renka'
        try:
            # 1. Find the TR for the horse number
            # The Waku/Umaban cell usually has the horse number as text.
            # XPath: Find tr that has a td with class Waku* or simple td containing exact text.
            # v9 fix: Use strict text matching for horse number to avoid partial matches (e.g. '1' matching '10')
            
            row_xpath = f"//tr[td[normalize-space(text())='{horse_num}']]"
            
            # 2. Find the label inside the row
            # The structure is td.Vote > ul > li.{mark_type} > label
            label_xpath = f"{row_xpath}//td[contains(@class, 'Vote')]//li[contains(@class, '{mark_type}')]//label"
            
            elem = driver.find_element(By.XPATH, label_xpath)
            
            # Check if already selected (class 'Selected')
            if "Selected" not in elem.get_attribute("class"):
                elem.click()
                time.sleep(0.1) # Small delay
                
        except Exception as e:
            logger.warning(f"[{race_id}] Failed to mark {mark_type} for {horse_num}: {e}")

    # Set marks
    click_mark(marks['honmei'], "Honmei")
    
    for h in marks['taiko']:
        click_mark(h, "Taiko")
    for h in marks['tanana']:
        click_mark(h, "Tanana")
    for h in marks['renka']:
        click_mark(h, "Renka")
        
    save_evidence(driver, race_id, "02_marks_set")

def go_to_betting_page(driver, race_id):
    """
    Transitions to the IPAT betting page.
    v9 Update: Uses direct navigation or ensures 'IPAT Vote' button works by waiting.
    """
    logger.info(f"[{race_id}] Click IPAT button...")
    wait = WebDriverWait(driver, 10)
    
    try:
        # Click 'IPAT' button (Vote Input)
        btn = wait.until(EC.element_to_be_clickable((By.ID, "act-ipat")))
        btn.click()
        
        # Handle Confirm Modal if any
        try:
            WebDriverWait(driver, 3).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            logger.info(f"[{race_id}] Alert: {alert.text}")
            alert.accept()
        except TimeoutException:
            pass
            
        # Wait for URL transition
        wait.until(EC.url_contains("ipat_sp.html"))
        logger.info(f"[{race_id}] Transistion to Betting Page COMPLETE.")
        
        # v9: CRITICAL WAIT for the Ticket Type buttons to appear.
        # This confirms that the Riot app has loaded and fetched race data.
        # Selector: div.RaceOdds_Menu01 or ul.Col4
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.Col4 li")))
            logger.info(f"[{race_id}] IPAT Ticket Types Loaded.")
        except TimeoutException:
            logger.error(f"[{race_id}] Betting Interface NOT loaded (Ticket Types missing).")
            save_evidence(driver, race_id, "04_ipat_failed_load")
            return False
            
        save_evidence(driver, race_id, "04_ipat_page")
        # dump_dom(driver, race_id, "ipat_dom") # Included in save_evidence
        return True
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to go to betting page: {e}")
        save_evidence(driver, race_id, "03_transition_fail")
        return False

def place_bets_logic(driver, race_id, bets):
    """
    Inputs bets into the OrePro IPAT interface.
    """
    logger.info(f"[{race_id}] input bets...")
    wait = WebDriverWait(driver, 5)
    
    type_map = {
        "単勝": "単勝", "複勝": "複勝", "枠連": "枠連", 
        "馬連": "馬連", "ワイド": "ワイド", "馬単": "馬単", 
        "3連複": "3連複", "3連単": "3連単"
    }
    
    for i, bet in enumerate(bets):
        try:
            logger.info(f"[{race_id}] Betting Loop {i+1} START")
            b_type_str = bet['type'] # e.g. "ワイド"
            b_type_btn_text = type_map.get(b_type_str, b_type_str)
            
            # 1. Select Ticket Type
            # v9 Update: The buttons are likely inside ul.Col4 as li elements.
            # We need to click the li that contains the text.
            try:
                # Find all li in Col4
                type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
                target_btn = None
                for btn in type_btns:
                    if b_type_btn_text in btn.text:
                        target_btn = btn
                        break
                
                if target_btn:
                    target_btn.click()
                else:
                    raise NoSuchElementException(f"Button for {b_type_btn_text} not found")
                    
            except Exception as e:
                logger.error(f"[{race_id}] Failed to find type btn for {b_type_str}")
                continue # Skip this bet
            
            time.sleep(0.5)
            
            # 2. Select Horses
            # Depending on 'BOX', 'SINGLE' etc?
            # User report says: "ワイド BOX: 14-11-1-10 BOX"
            # OrePro UI usually has "Method" selection? 
            # Or standard IPAT just selects horses and you click "Box" button/tab?
            # OrePro IPAT (from DOM) seems to have simple "Select Horses" list.
            # And maybe a "Box" button isn't visible in the DOM dump?
            # Wait, DOM dump showed: <div class="Race_Odds_Menu_Title">券種選択</div>
            
            # Let's assume standard flow: Select Type -> Select Horses -> Add to Cart.
            # If it's a BOX bet, usually one selects multiple horses and creates a box?
            # Or does OrePro handle "Box" as a separate mode?
            
            # For now, let's just click the horses. 
            # If logic requires "Form" selection (Nagashi/Box), we need to find that context.
            # But standard OrePro IPAT (Simple) might just be Multi-Select -> Add?
            # Let's look at `ipat_dom.html` line 96... it just shows horse list.
            
            for h in bet['horses']:
                # Find current state of the horse button
                # Table row tr with horse num?
                # Selector: table.RaceOdds_HorseList_Table tr td.Waku(text=h) -> parent tr -> click
                # Or click the row?
                try:
                    # v9: Robust row selector
                    row_xpath = f"//table[contains(@class, 'RaceOdds_HorseList_Table')]//tr[td[normalize-space(text())='{h}']]"
                    row = driver.find_element(By.XPATH, row_xpath)
                    
                    # Check if already selected?
                    # The DOM style says: tr.selected { background: ... }
                    classes = row.get_attribute("class")
                    if "selected" not in classes:
                        row.click()
                        
                except Exception as e:
                    logger.warning(f"[{race_id}] Failed to select horse {h}: {e}")
            
            time.sleep(0.5)
            
            # 3. Input Amount
            # Selector: input[name="money"]
            try:
                money_input = driver.find_element(By.NAME, "money")
                money_input.clear()
                # Amount is in 100 yen units?
                # DOM: `input ...` followed by `00円`.
                # If bet['amount'] is 600, input 6?
                # User report: "600円". So input 6.
                amount_val = int(bet['amount']) // 100
                money_input.send_keys(str(amount_val))
            except Exception as e:
                logger.error(f"[{race_id}] Failed to input money: {e}")
            
            # 4. Click Add (追加)
            try:
                add_btn = driver.find_element(By.XPATH, "//button[contains(text(), '追加')]")
                add_btn.click()
            except Exception as e:
                logger.error(f"[{race_id}] Failed to click Add: {e}")
            
            time.sleep(1.0)
            
            # Handle potential alerts ("Box/Formation" confirmation?)
            try:
                alert = driver.switch_to.alert
                logger.info(f"[{race_id}] Alert: {alert.text}")
                alert.accept()
            except:
                pass
                
            save_evidence(driver, race_id, f"05_bet_{i+1}_added")
            
        except Exception as e:
            logger.error(f"[{race_id}] Betting loop error: {e}")
            save_evidence(driver, race_id, "betting_error")

    # 5. Finalize (Set Bets) - "買い目をセットする"
    logger.info(f"[{race_id}] Setting bets...")
    try:
        set_btn = driver.find_element(By.CLASS_NAME, "SetBtn")
        set_btn.click()
        
        # This usually returns to Shutuba page or updates state?
        # Wait for "Finalize Vote" button on parent page?
        # Or does it redirect?
        # In `v8` logs, next step was "Finalizing...".
        # If `SetBtn` closes the "IPAT" modal/page, we return to `shutuba`.
        # Then on `shutuba`, we click "投票する" (Vote)?
        
        # Wait for transition back or update
        time.sleep(5)
        save_evidence(driver, race_id, "07_post_set")
        
        # 6. Click Final Vote Button
        # Button: "この予想で勝負！" (act-bet_...)
        logger.info(f"[{race_id}] Finalizing...")
        final_btn = wait.until(EC.element_to_be_clickable((By.ID, f"act-bet_{race_id}")))
        final_btn.click()
        save_evidence(driver, race_id, "08_final_btn_view")
        
        # Confirm final alert
        try:
            WebDriverWait(driver, 5).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            logger.info(f"[{race_id}] Final Alert: {alert.text}")
            alert.accept()
        except:
            pass
            
        # Check for completion (redirect to complete page or message)
        time.sleep(5)
        if "complete" in driver.current_url or "リクエストを受け付けました" in driver.page_source:
             logger.info(f"[{race_id}] VOTE SUCCESS.")
        else:
             logger.warning(f"[{race_id}] Warning: No completion page detected. URL: {driver.current_url}")
             save_evidence(driver, race_id, "09_vote_result")

    except Exception as e:
        logger.error(f"[{race_id}] Finalize error: {e}")
        save_evidence(driver, race_id, "finalize_error")

def main():
    # Load secrets
    secrets = load_secrets()
    
    # Parse Report
    bets_by_race = parse_report()
    
    # Read report content for marks
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    marks_by_race = parse_marks(content)
    
    logger.info(f"Parsed {len(bets_by_race)} races from report.")
    
    # Setup Driver
    driver = setup_driver()
    
    try:
        # Login
        login_netkeiba(driver, secrets)
        
        # Process each race
        for race_id in sorted(bets_by_race.keys()):
            logger.info(f"\n--- Processing {race_id} ---")
            
            # Verify bets exist
            if not bets_by_race[race_id]:
                logger.warning(f"[{race_id}] No bets found in data!")
                continue
            logger.info(f"[{race_id}] Found {len(bets_by_race[race_id])} bets.")
            
            # 1. Ensure Page
            if not ensure_shutuba_page(driver, race_id):
                continue
            
            # 2. Set Marks (REQUIRED for IPAT)
            if race_id in marks_by_race:
                set_prediction_marks(driver, race_id, marks_by_race[race_id])
            else:
                logger.warning(f"[{race_id}] No marks found.")
            
            # 3. Go to Betting Page
            if go_to_betting_page(driver, race_id):
                # 4. Place Bets
                place_bets_logic(driver, race_id, bets_by_race[race_id])
            
            time.sleep(1) # Interval
            
    except Exception as e:
        logger.error(f"Main loop error: {traceback.format_exc()}")
    finally:
        driver.quit()
        logger.info("Detailed logs and screenshots saved to " + LOG_DIR)

if __name__ == "__main__":
    main()
