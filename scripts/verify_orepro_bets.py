
import json
import re
import time
import os
import sys
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration ---
REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/verification_evidence"

# Setup Logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "verification.log"),
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
        sys.exit(1)

def parse_race_ids():
    """Extract race IDs from report."""
    race_ids = []
    venue_map = {"東京": "05", "京都": "08", "小倉": "10"}
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            m_race = re.search(r"## (東京|京都|小倉)(\d+)R", line)
            if m_race:
                venue_name = m_race.group(1)
                race_num = int(m_race.group(2))
                if venue_name == "東京": base = "2026050103"
                elif venue_name == "京都": base = "2026080203"
                elif venue_name == "小倉": base = "2026100105"
                race_ids.append(f"{base}{race_num:02d}")
    return race_ids

def verify_race(driver, race_id):
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
    driver.get(url)
    time.sleep(2)
    
    status = "UNKNOWN"
    details = ""
    
    try:
        # Check Kaime List
        kaime_list = driver.find_elements(By.CSS_SELECTOR, "ul.Kaime_List li")
        
        if len(kaime_list) > 0:
            status = "OK"
            details = f"Bets found: {len(kaime_list)}"
            
            # Try to get amount
            try:
                money_elem = driver.find_element(By.CSS_SELECTOR, "div.BakenMoney")
                details += f" | {money_elem.text.replace('\n', ' ')}"
            except:
                pass
                
        else:
            status = "EMPTY"
            details = "No bets found in Kaime_List."
            
        # Valid marks check (Honmei)
        honmei_labels = driver.find_elements(By.CSS_SELECTOR, "li.Honmei label.Selected")
        if honmei_labels:
            details += f" | Honmei Mark: Yes"
        else:
            details += f" | Honmei Mark: NO"
            if status == "OK": status = "WARNING (Bets exist but no Honmei?)"

        # Save screenshot
        driver.save_screenshot(os.path.join(LOG_DIR, f"{race_id}_verify.png"))
        
    except Exception as e:
        status = "ERROR"
        details = str(e)
        
    logger.info(f"[{race_id}] {status} - {details}")
    return status, details

def main():
    secrets = load_secrets()
    race_ids = parse_race_ids()
    logger.info(f"Verifying {len(race_ids)} races...")
    
    driver = setup_driver()
    try:
        login_netkeiba(driver, secrets)
        
        results = []
        for race_id in sorted(race_ids):
            s, d = verify_race(driver, race_id)
            results.append((race_id, s, d))
            
        # Summary
        logger.info("\n=== VERIFICATION SUMMARY ===")
        ok_count = sum(1 for r in results if r[1] == "OK")
        logger.info(f"Total: {len(results)}, OK: {ok_count}, Issues: {len(results) - ok_count}")
        
        for r in results:
            if r[1] != "OK":
                logger.warning(f"{r[0]}: {r[1]} - {r[2]}")
                
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
