import json
import time
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# --- 設定 ---
SECRETS_FILE = "scripts/debug/netkeiba_secrets.json"

# 小倉 8R - 12R のデータ (ID修正: Day 5 for Feb 7)
# Correct for Feb 7: 2026100105xx (1回小倉5日)
RACES = [
    {
        "race_id": "202610010508",
        "name": "小倉8R",
        "marks": {2: 1, 1: 2, 3: 3, 10: 4},  # 馬番: 印index (1:◎, 2:○, 3:▲, 4:△)
        "bets": [
            {"type": "wide_box", "horses": [2, 1, 3, 10], "amount": 100},
            {"type": "win", "horse": 2, "amount": 200},
            {"type": "win", "horse": 1, "amount": 100},
            {"type": "win", "horse": 3, "amount": 100}
        ]
    },
    {
        "race_id": "202610010509",
        "name": "小倉9R",
        "marks": {8: 1, 11: 2, 2: 3, 1: 4},
        "bets": [
            {"type": "wide_box", "horses": [8, 11, 2, 1], "amount": 100},
            {"type": "win", "horse": 8, "amount": 200},
            {"type": "win", "horse": 11, "amount": 100},
            {"type": "win", "horse": 2, "amount": 100}
        ]
    },
    {
        "race_id": "202610010510",
        "name": "小倉10R",
        "marks": {6: 1, 1: 2, 13: 3, 16: 4},
        "bets": [
            {"type": "wide_box", "horses": [6, 1, 13, 16], "amount": 100},
            {"type": "win", "horse": 6, "amount": 200},
            {"type": "win", "horse": 1, "amount": 100},
            {"type": "win", "horse": 13, "amount": 100}
        ]
    },
    {
        "race_id": "202610010511",
        "name": "小倉11R",
        "marks": {1: 1, 13: 2, 14: 3, 7: 4},
        "bets": [
            {"type": "wide_box", "horses": [1, 13, 14, 7], "amount": 100},
            {"type": "win", "horse": 1, "amount": 200},
            {"type": "win", "horse": 13, "amount": 100},
            {"type": "win", "horse": 14, "amount": 100}
        ]
    },
    {
        "race_id": "202610010512",
        "name": "小倉12R",
        "marks": {16: 1, 13: 2, 1: 3, 4: 4},
        "bets": [
            {"type": "wide_box", "horses": [16, 13, 1, 4], "amount": 100},
            {"type": "win", "horse": 16, "amount": 200},
            {"type": "win", "horse": 13, "amount": 100},
            {"type": "win", "horse": 1, "amount": 100}
        ]
    }
]

def load_secrets():
    try:
        with open(SECRETS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {SECRETS_FILE} not found.")
        return {}

def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1280,1024")
    # Headless toggle (Default off for Windows usually, but on server...)
    # options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

def safe_click(driver, by, val, timeout=5):
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, val))).click()
        return True
    except:
        try:
            elem = driver.find_element(by, val)
            driver.execute_script("arguments[0].click();", elem)
            return True
        except:
            return False

def login(driver, secrets):
    print("--- Login ---")
    driver.get("https://regist.netkeiba.com/account/?pid=login")
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "login_id")))
        driver.find_element(By.NAME, "login_id").send_keys(secrets.get('email', ''))
        driver.find_element(By.NAME, "pswd").send_keys(secrets.get('password', ''))
        driver.find_element(By.CSS_SELECTOR, "input[alt='ログイン']").click()
        WebDriverWait(driver, 15).until(EC.url_changes("https://regist.netkeiba.com/account/?pid=login"))
        print("Login success.")
    except Exception as e:
        print(f"Login failed or already logged in: {e}")

def process_race(driver, race):
    rid = race['race_id']
    name = race['name']
    print(f"\nProcessing {name} ({rid})...")

    # 1. Shutuba
    url = f"https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={rid}"
    driver.get(url)
    
    # Wait for table
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "Umaban")))
    except:
        print(f"Failed to load shutuba page for {name} (URL: {url})")
        return

    # Set Marks
    print(f"  Setting marks...")
    for horse_num, mark_idx in race['marks'].items():
        try:
            xpath = f"//tr[.//td[contains(@class, 'Umaban') and normalize-space(text())='{horse_num}']]//td[contains(@class, 'Mark')]//label[{mark_idx}]"
            elems = driver.find_elements(By.XPATH, xpath)
            if elems:
                driver.execute_script("arguments[0].click();", elems[0])
            else:
                print(f"    Mark label not found for horse {horse_num}")
        except Exception as e:
            print(f"    Error setting mark for horse {horse_num}: {e}")

    # 2. IPAT
    print("  Moving to IPAT...")
    if not safe_click(driver, By.ID, "act-ipat"):
        print("    Failed to click act-ipat")
        return
    
    # Modal
    try:
        WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-orange"))).click()
    except:
        pass

    # IPAT Page
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "shikibetsu_btn")))
    except:
        print("    Failed to load IPAT page")
        return

    # 3. Enter Bets
    for bet in race['bets']:
        print(f"  Betting: {bet['type']} {bet.get('horses') or bet.get('horse')} {bet['amount']}円")
        
        # Select Type
        if bet['type'] == 'wide_box':
            safe_click(driver, By.XPATH, "//a[contains(@class,'shikibetsu_btn') and contains(text(),'ワイド')]")
            time.sleep(0.5)
            safe_click(driver, By.XPATH, "//a[contains(text(),'ボックス')]")
        elif bet['type'] == 'win':
            safe_click(driver, By.XPATH, "//a[contains(@class,'shikibetsu_btn') and contains(text(),'単勝')]")
            time.sleep(0.5)
            safe_click(driver, By.XPATH, "//a[contains(text(),'通常')]")
        
        time.sleep(0.5)

        # Clear
        driver.execute_script("document.querySelectorAll('label.Check01Btn_On').forEach(el => el.click());")
        time.sleep(0.2)

        # Select Horses
        horses = bet.get('horses', [bet.get('horse')])
        for h in horses:
            # tr id=tr_{h}
            xpath = f"//tr[@id='tr_{h}']//label[contains(@class, 'Check')]"
            if not safe_click(driver, By.XPATH, xpath):
                xpath2 = f"//tr[.//td[contains(@class,'Umaban') and normalize-space(text())='{h}']]//label[contains(@class,'Check')]"
                safe_click(driver, By.XPATH, xpath2)

        # Input Money
        try:
            inp = driver.find_element(By.NAME, "money")
            inp.clear()
            inp.send_keys(str(bet['amount'] // 100))
            driver.execute_script("arguments[0].dispatchEvent(new Event('input', {bubbles:true}));", inp)
            driver.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles:true}));", inp)
        except Exception as e:
            print(f"    Money input failed: {e}")

        # Add
        if not safe_click(driver, By.XPATH, "//button[contains(text(), '追加')]"):
            print("    Failed to click Add")
        
        time.sleep(1)

    # 4. Submit
    print("  Submitting...")
    if safe_click(driver, By.CSS_SELECTOR, "button.SetBtn"):
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, f"act-bet_{rid}"))).click()
            WebDriverWait(driver, 10).until(EC.url_contains("bet_complete.html"))
            print(f"  {name} COMPLETE!")
        except Exception as e:
            print(f"  Submit failed: {e}")

    time.sleep(2)

def main():
    secrets = load_secrets()
    if not secrets:
        print("No secrets found check file path.")
        return

    driver = setup_driver()
    try:
        login(driver, secrets)
        for race in RACES:
            process_race(driver, race)
    finally:
        print("Closing driver...")
        driver.quit()

if __name__ == "__main__":
    main()
