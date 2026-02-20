import time
import os
import sys
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'netkeiba_secrets.json')

def load_secrets():
    with open(SECRETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

options = Options()
# ブラウザを表示して動作確認
# options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
secrets = load_secrets()

print("Logging in...")
driver.get('https://regist.netkeiba.com/account/?pid=login')
time.sleep(2)
driver.find_element(By.NAME, 'login_id').send_keys(secrets['email'])
driver.find_element(By.NAME, 'pswd').send_keys(secrets['password'])
try:
    btn = driver.find_element(By.XPATH, "//input[@type='image' and contains(@alt, 'ログイン')]")
except Exception:
    btn = driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
btn.click()
time.sleep(3)

print("Navigating to race...")
driver.get('https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id=202605010701')
time.sleep(4)
print("Clicking '買い目を入力する'...")
btn = driver.find_element(By.ID, 'act-ipat')
driver.execute_script('arguments[0].click();', btn)
time.sleep(4)

print("Waiting for IPAT to load...")
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

try:
    WebDriverWait(driver, 10).until(lambda d: 'ipat_sp' in d.current_url or len(d.window_handles) > 1)
except TimeoutException:
    print("Timeout waiting for IPAT page.")

if len(driver.window_handles) > 1:
    driver.switch_to.window(driver.window_handles[-1])

print("Waiting for elements...")
try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.Col4 li')))
except:
    pass
time.sleep(3)

print("Getting DOM and Screenshot...")
html = driver.page_source
log_file = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'ipat_dom_investigation.html')
screenshot_file = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'ipat_screenshot.png')

with open(log_file, 'w', encoding='utf-8') as f:
    f.write(html)
driver.save_screenshot(screenshot_file)

print(f"Saved DOM to {log_file}")
print(f"Saved Screenshot to {screenshot_file}")

driver.quit()
