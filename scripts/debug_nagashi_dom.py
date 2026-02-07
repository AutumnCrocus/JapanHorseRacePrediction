
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

# --- Configuration ---
NETKEIBA_SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"
LOG_DIR = "scripts/debug/nagashi_analysis"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "nagashi_debug.log"),
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

def save_evidence(driver, name):
    try:
        path_img = os.path.join(LOG_DIR, f"{name}.png")
        driver.save_screenshot(path_img)
        logger.info(f"📸 Saved screenshot: {path_img}")
        path_dom = os.path.join(LOG_DIR, f"{name}_dom.html")
        with open(path_dom, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")

def login_netkeiba(driver, secrets):
    logger.info("[LOGIN] ...")
    driver.get("https://regist.netkeiba.com/account/?pid=login")
    wait = WebDriverWait(driver, 10)
    try:
        user_input = wait.until(EC.presence_of_element_located((By.NAME, "login_id")))
        pass_input = driver.find_element(By.NAME, "pswd")
        user_input.send_keys(secrets['email'])
        pass_input.send_keys(secrets['password'])
        driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')] | //input[@type='image' and contains(@alt, 'ログイン')]").click()
        wait.until(EC.url_contains("netkeiba.com"))
        logger.info("[LOGIN] Success.")
    except Exception as e:
        logger.error(f"[LOGIN] Failed: {e}")
        sys.exit(1)

def handle_popups(driver):
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
    except: pass
    try:
        for btn in driver.find_elements(By.XPATH, "//button[contains(text(), 'はい')]"):
            if btn.is_displayed(): btn.click()
    except: pass

def main():
    secrets = load_secrets()
    driver = setup_driver()
    
    # Target: Kokura 9R (202610010609) 3-Renpuku Nagashi
    race_id = "202610010609"
    target_type = "3連複"
    target_method = "ながし"

    try:
        login_netkeiba(driver, secrets)
        
        # Go to Shutuba
        url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}"
        driver.get(url)
        time.sleep(3)
        handle_popups(driver)
        
        # Click Input
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
            logger.error("Input button not found.")
            return

        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_btn)
        time.sleep(0.5)
        try:
            input_btn.click()
        except ElementClickInterceptedException:
            handle_popups(driver)
            driver.execute_script("arguments[0].click();", input_btn)
        
        time.sleep(2)
        handle_popups(driver)
        
        # Precise wait for transition
        try:
            WebDriverWait(driver, 10).until(lambda d: "ipat_sp" in d.current_url or len(d.window_handles) > 1)
        except TimeoutException:
            logger.error("Failed transition to IPAT (Timeout).")
            # Try once more with JS click
            try:
                driver.execute_script("arguments[0].click();", input_btn)
                WebDriverWait(driver, 10).until(lambda d: "ipat_sp" in d.current_url or len(d.window_handles) > 1)
            except:
                logger.error("Retry failed too.")
                save_evidence(driver, "ipat_transition_fail")
                return

        if len(driver.window_handles) > 1: 
            driver.switch_to.window(driver.window_handles[-1])
            logger.info(f"Switched to window: {driver.current_url}")
        else:
            logger.info(f"Stayed in window: {driver.current_url}")
            
        logger.info("--- On IPAT Page ---")
        
        # Wait for Riot.js to render
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.Col4 li")))
            time.sleep(1) # Extra buffer
        except TimeoutException:
            logger.error("Timeout waiting for buttons to render.")
            save_evidence(driver, "riot_render_fail")
            return

        # Debug: List all type buttons
        type_btns = driver.find_elements(By.CSS_SELECTOR, "ul.Col4 li")
        logger.info(f"Found {len(type_btns)} type buttons.")
        for i, btn in enumerate(type_btns):
            logger.info(f"Btn {i}: '{btn.text}'")
            if target_type in btn.text:
                btn.click()
                logger.info(f"Clicked {target_type}")
                time.sleep(1) # Wait for Method menu to appear?
                break
        else:
            logger.error(f"Target type '{target_type}' not found in buttons.")
            save_evidence(driver, "type_click_fail")
            return

        # Select Method: ながし
        # Method area might depend on type selection
        # Actually search for "方式選択" text
        try:
            # Wait for method label?
            time.sleep(1) 
            method_label = driver.find_element(By.XPATH, "//div[contains(text(), '方式選択')]")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", method_label)
            method_list = method_label.find_elements(By.XPATH, "./following-sibling::ul//li")
            logger.info(f"Found {len(method_list)} method buttons.")
            
            for m in method_list:
                logger.info(f"Method Btn: '{m.text}'")
                if target_method in m.text:
                    m.click()
                    logger.info(f"Clicked {target_method}")
                    break
            else:
                logger.error(f"Target method '{target_method}' not found.")
        except Exception as e:
            logger.error(f"Method selection error: {e}")

        time.sleep(3) # Wait for UI update
        
        # Inspect DOM for '軸' and '相手'
        page_source = driver.page_source
        if "軸" in page_source:
            logger.info("Found '軸' in DOM!")
            # Find elements containing '軸'
            axis_els = driver.find_elements(By.XPATH, "//*[contains(text(), '軸')]")
            for el in axis_els:
                # Get text safely
                txt = el.text
                p_tag = el.find_element(By.XPATH, '..').tag_name
                logger.info(f"Axis Element: {el.tag_name} - {txt} - Parent: {p_tag}")
        else:
            logger.warning("'軸' NOT found in DOM.")
            
        if "相手" in page_source:
             logger.info("Found '相手' in DOM!")
             partner_els = driver.find_elements(By.XPATH, "//*[contains(text(), '相手')]")
             for el in partner_els:
                 logger.info(f"Partner Element: {el.tag_name} - {el.text}")
        
        # Capture DOM!!
        save_evidence(driver, "kokura09r_nagashi_ui_v5")
        
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
