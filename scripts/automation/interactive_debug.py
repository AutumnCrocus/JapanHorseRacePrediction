import time
import os
import sys
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'netkeiba_secrets.json')

def load_secrets():
    with open(SECRETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

options = Options()
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

print("Clicking target horse (2)...")
xpath = f"//tr[contains(@class, 'HorseList') and td[text()='2']]"
row = driver.find_element(By.XPATH, xpath)
row_id = row.get_attribute('id')
internal_id = row_id.replace('tr_', '')
mark_id = f'm1-{internal_id}'
label_id = f'ml1-{internal_id}'
input_el = driver.find_element(By.ID, mark_id)
if not input_el.is_selected():
    label_btn = driver.find_element(By.ID, label_id)
    driver.execute_script('arguments[0].click();', label_btn)
    time.sleep(1)

print("Finding and clicking the '買い目を入力する' button...")
input_btn = None
for sel_type, sel_val in [
    (By.ID, 'act-ipat'),
    (By.XPATH, "//button[contains(text(), '買い目を入力する')]"),
    (By.XPATH, "//a[contains(text(), '買い目を入力する')]"),
]:
    try:
        input_btn = driver.find_element(sel_type, sel_val)
        if input_btn.is_displayed():
            break
    except Exception:
        continue

if input_btn:
    print(f"Found input_btn using {sel_type}: {sel_val}")
    driver.execute_script('arguments[0].scrollIntoView({block: "center"});', input_btn)
    time.sleep(1)
    driver.execute_script('arguments[0].click();', input_btn)
else:
    print("Could not find input_btn")

print("Waiting for IPAT transition...")
import code
# 対話モードに入って現在のブラウザの状態を確認できるようにする
code.interact(local=locals())

driver.quit()
