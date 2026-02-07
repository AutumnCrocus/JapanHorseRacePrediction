# 設定
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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException, NoAlertPresentException, NoSuchElementException

# --- 設定 ---
REPORT_FILE = "prediction_report_20260207_hybrid.md"
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
KAISAI_DATE = "20260207"
KAISAI_IDS = {
    "東京": "2026050103",
    "京都": "2026080203",
    "小倉": "2026100105"
}

# ログ/スクショ保存先
DEBUG_DIR = "scripts/debug/screenshots_v8"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEBUG_DIR, "automation_v8.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def save_evidence(driver, race_id, step_name):
    timestamp = datetime.now().strftime("%H%M%S")
    base_name = f"{race_id}_{timestamp}_{step_name}"
    png_path = os.path.join(DEBUG_DIR, f"{base_name}.png")
    try:
        driver.save_screenshot(png_path)
        logger.info(f"📸 Saved screenshot: {png_path}")
    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")

def dump_dom(driver, race_id, step_name):
    """DOMダンプ（デバッグ用）"""
    try:
        html = driver.page_source
        path = os.path.join(DEBUG_DIR, f"{race_id}_{step_name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"📄 Saved DOM: {path}")
    except Exception as e:
        logger.error(f"Failed to save DOM: {e}")

def load_netkeiba_secrets():
    try:
        with open(NETKEIBA_SECRETS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: {NETKEIBA_SECRETS_FILE} not found.")
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
    logger.info(f"Parsed {len(bets_by_race)} races from report.")
    return bets_by_race

def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("--excludeSwitches=enable-logging")
    options.add_argument("--window-size=1280,1024")
    driver = webdriver.Chrome(options=options)
    return driver

def safe_click(driver, selector_type, selector_value, timeout=5): 
    wait = WebDriverWait(driver, timeout)
    try:
        elem = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        time.sleep(0.3) 
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
    logger.info("[LOGIN] Acccessing login page...")
    
    try:
        try:
            WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CLASS_NAME, "Header_User_Icon")))
            logger.info("[LOGIN] Already logged in.")
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
        logger.info("[LOGIN] Success.")
        
    except Exception as e:
        logger.error(f"[LOGIN] Error: {e}")
        sys.exit(1)

def ensure_shutuba_page(driver, race_id):
    """page check logic (relaxed)"""
    max_retries = 2
    for attempt in range(max_retries):
        current_url = driver.current_url
        if "shutuba.html" in current_url and "mode=init" in current_url:
            if len(driver.find_elements(By.ID, "act-ipat")) > 0:
                logger.info(f"[{race_id}] Verified Start List page (button found).")
                return True
            if len(driver.find_elements(By.TAG_NAME, "tr")) > 5:
                logger.info(f"[{race_id}] Verified Start List page (rows found).")
                return True
        
        url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
        logger.info(f"[{race_id}] Navigating to: {url}")
        driver.get(url)
        time.sleep(2)
        
        if len(driver.find_elements(By.ID, "act-ipat")) > 0:
             logger.info(f"[{race_id}] Verified Start List page after navigation.")
             save_evidence(driver, race_id, "01_shutuba_ok")
             return True
        
    logger.error(f"[{race_id}] Could not reach Shutuba page (or button missing).")
    save_evidence(driver, race_id, f"01_shutuba_fail_final")
    return False

def set_prediction_marks(driver, race_id, bets):
    """予想印を設定する"""
    if not ensure_shutuba_page(driver, race_id):
        return
    
    # Dump DOM for debugging selector
    dump_dom(driver, race_id, "shutuba_dom")

    all_horses = []
    for bet in bets:
        all_horses.extend(bet['horses'])
    
    unique_horses = []
    found_horses = set()
    for h in all_horses:
        if h not in found_horses:
            unique_horses.append(h)
            found_horses.add(h)
    
    if not unique_horses:
        return

    honmei = unique_horses[0]
    others = unique_horses[1:]
    
    logger.info(f"[{race_id}] Marking: ◎={honmei}, ○={others}")
    
    try:
        # ◎ - Try generic xpath (any clickable child)
        honmei_row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{honmei}']]//td[contains(@class, 'Mark')]//*[contains(@class, 'Mark') or contains(@class, 'Btn') or tag_name='label' or tag_name='span'][1]"
        # Fallback to direct child
        honmei_direct_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{honmei}']]//td[contains(@class, 'Mark')]/*[1]"
        
        if not safe_click(driver, By.XPATH, honmei_row_xpath, timeout=2):
             if not safe_click(driver, By.XPATH, honmei_direct_xpath, timeout=2):
                logger.warning(f"[{race_id}] Failed to mark Honmei {honmei}")
        
        # ○
        for h in others:
            row_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{h}']]//td[contains(@class, 'Mark')]//*[2]"
             # Fallback to direct child
            row_direct_xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{h}']]//td[contains(@class, 'Mark')]/*[2]"
            
            if not safe_click(driver, By.XPATH, row_xpath, timeout=1):
                safe_click(driver, By.XPATH, row_direct_xpath, timeout=1)
        
        save_evidence(driver, race_id, "02_marks_set")
            
    except Exception as e:
        logger.error(f"[{race_id}] Error setting marks: {e}")

    # 「買い目を入力する」ボタンへ
    go_to_betting_page(driver, race_id)

def go_to_betting_page(driver, race_id):
    """投票画面へ遷移"""
    logger.info(f"[{race_id}] Click IPAT button...")
    
    if not safe_click(driver, By.ID, "act-ipat"):
        logger.error(f"[{race_id}] Failed to click act-ipat.")
        return

    # モーダル処理
    try:
        modal_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-orange"))
        )
        if "はい" in modal_btn.text:
            modal_btn.click()
            logger.info(f"[{race_id}] Handled confirmation modal.")
    except TimeoutException:
        pass
    except Exception as e:
        logger.error(f"[{race_id}] Modal error: {e}")

    try:
        WebDriverWait(driver, 10).until(EC.url_contains("ipat_sp.html"))
        logger.info(f"[{race_id}] Transistion to Betting Page COMPLETE.")
        save_evidence(driver, race_id, "04_ipat_page")
        
        # Dump DOM for IPAT debugging
        dump_dom(driver, race_id, "ipat_dom")
        
    except Exception as e:
        logger.error(f"[{race_id}] Failed to reach betting page: {e}")

def place_bets_logic(driver, race_id, bets):
    """IPAT画面での入力処理"""
    wait = WebDriverWait(driver, 10)
    try:
        # Check for ANY shikibetsu btn, or list
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
    except Exception as e:
        logger.error(f"[{race_id}] TIMEOUT looking for 'shikibetsu_btn'. URL: {driver.current_url}")
        save_evidence(driver, race_id, "05_ipat_timeout")
        return

    for i, bet in enumerate(bets):
        try:
            logger.info(f"[{race_id}] Betting Loop {i+1} START")
            # 券種
            if not safe_click(driver, By.XPATH, f"//li[contains(@class, 'shikibetsu_btn') and contains(text(), '{bet['type']}')]"):
                logger.error(f"[{race_id}] Failed to find type btn for {bet['type']}")
                continue
            
            # 方式
            if bet['type'] not in ['単勝', '複勝']:
                method_map = {'SINGLE': '通常', 'BOX': 'ボックス', '流し': 'ながし'}
                t = method_map.get(bet['method'], '通常')
                time.sleep(0.3)
                safe_click(driver, By.XPATH, f"//a[contains(text(), '{t}')]")

            # 馬番選択
            for h in bet['horses']:
                try:
                    lbl = driver.find_element(By.XPATH, f"//label[contains(@class, 'Check01Btn') and normalize-space(text())='{h}']")
                    if "Check01Btn_Off" in lbl.get_attribute("class"):
                        lbl.click()
                except:
                    pass

            time.sleep(0.5)

            # 金額 (1=100円)
            try:
                inp = driver.find_element(By.NAME, "money")
                inp.clear()
                coins = max(1, bet['amount'] // 100)
                inp.send_keys(str(coins))
            except:
                pass

            # 追加
            driver.find_element(By.XPATH, "//button[contains(text(), '追加')]").click()
            
            time.sleep(1.0) 
            
            # アラートハンドリング
            try:
                alert = driver.switch_to.alert
                alert.accept()
            except NoAlertPresentException:
                pass
            
            save_evidence(driver, race_id, f"05_bet_{i+1}_added")
                
        except Exception as e:
            logger.error(f"[{race_id}] Loop error: {e}")

    # セットして戻る
    logger.info(f"[{race_id}] Setting bets...")
    safe_click(driver, By.CSS_SELECTOR, "button.SetBtn")
    
    try:
        wait.until(EC.url_contains("shutuba.html"))
        save_evidence(driver, race_id, "07_post_set")
    except:
        logger.error(f"[{race_id}] Failed to return to shutuba.")

def finalize_race(driver, race_id):
    """最終投票ボタン押下"""
    logger.info(f"[{race_id}] Finalizing...")
    btn_id = f"act-bet_{race_id}"
    wait = WebDriverWait(driver, 10)
    
    try:
        wait.until(EC.presence_of_element_located((By.ID, btn_id)))
        elem = driver.find_element(By.ID, btn_id)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
        time.sleep(1)
        
        save_evidence(driver, race_id, "08_final_btn_view")
        elem.click()
        
        try:
            wait.until(EC.url_contains("bet_complete.html"))
            logger.info(f"[{race_id}] COMPLETE.")
            save_evidence(driver, race_id, "09_completed")
        except:
            logger.warning(f"[{race_id}] Warning: No completion page.")
    except Exception as e:
        logger.error(f"[{race_id}] Finalize error: {e}")

def main():
    secrets = load_netkeiba_secrets()
    bets_data = parse_report()
    driver = setup_driver()
    
    try:
        login_netkeiba(driver, secrets)
        
        race_ids = sorted(bets_data.keys())
        for race_id in race_ids:
            logger.info(f"\n--- Processing {race_id} ---")
            
            # Verify bets exist
            if not bets_data[race_id]:
                logger.warning(f"[{race_id}] No bets found in data!")
                continue
            logger.info(f"[{race_id}] Found {len(bets_data[race_id])} bets.")

            set_prediction_marks(driver, race_id, bets_data[race_id])
            
            if "ipat_sp.html" in driver.current_url:
                place_bets_logic(driver, race_id, bets_data[race_id])
                
                if "shutuba.html" in driver.current_url:
                    finalize_race(driver, race_id)
            
            time.sleep(1)
            
    finally:
        logger.info("Script finished.")
        # driver.quit()

if __name__ == "__main__":
    main()
