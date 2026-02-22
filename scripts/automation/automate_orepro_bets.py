import json
import re
import time
import os
import sys
import logging
import math
import argparse
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# ===== 設定 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECRETS_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'netkeiba_secrets.json')
LOG_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'screenshots_orepro_auto')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'orepro_auto.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger()
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logger.addHandler(console)

def load_secrets():
    with open(SECRETS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1280,1024')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(options=options)

def save_evidence(driver, race_id, step_name):
    try:
        ts = datetime.now().strftime('%H%M%S')
        path = os.path.join(LOG_DIR, f'{race_id}_{ts}_{step_name}.png')
        driver.save_screenshot(path)
        logger.info(f'📸 Screenshot: {path}')
    except Exception as e:
        logger.error(f'Screenshot failed: {e}')

def handle_popups(driver):
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
    except TimeoutException:
        pass
    for btn_sel in [
        (By.XPATH, "//button[contains(text(), 'はい')]"),
        (By.CSS_SELECTOR, '.swal-button--confirm'),
        (By.CSS_SELECTOR, '.jconfirm-buttons button'),
    ]:
        try:
            for btn in driver.find_elements(*btn_sel):
                if btn.is_displayed():
                    btn.click()
                    time.sleep(0.5)
                    return
        except Exception:
            pass

def check_error_popup(driver):
    try:
        for el in driver.find_elements(By.CSS_SELECTOR, '.swal-text'):
            if el.is_displayed() and el.text.strip():
                logger.error(f'Error Popup: {el.text}')
                return el.text
    except Exception:
        pass
    return None

def parse_report(report_path, target_type='catboost'):
    bets_by_race = {}
    
    if not os.path.exists(report_path):
        logger.error(f'レポートが見つかりません: {report_path}')
        return {}

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    races = re.split(r'\n## ', content)
    for race_block in races:
        if not race_block.strip():
            continue
            
        m_race = re.match(r'(.*?)\s*\((\d{12})\)', race_block)
        if not m_race: continue
        race_id = m_race.group(2)
        bets = []
        
        if target_type == 'catboost':
            catboost_sec = re.search(r'### 【CatBoost詳細分析.*?】(.*?)(?:\n---|\n## |\Z)', race_block, re.DOTALL)
            if not catboost_sec: continue
            
            cat_text = catboost_sec.group(1)
            if '平均EVが閾値' in cat_text and '未満のため' in cat_text:
                continue
                
            rec_sec = re.search(r'#### 買い目推奨:(.*?)(?:\n\n|\Z)', cat_text, re.DOTALL)
            if not rec_sec: continue
            
            for line in rec_sec.group(1).strip().split('\n'):
                line = line.strip()
                if not line.startswith('- **'): continue
                
                m = re.search(r'- \*\*([^\\*]+)\*\*: \[([\d, ]+)\] \((\d+)円\)', line)
                if not m: continue
                
                method_raw = m.group(1)
                horses_raw = m.group(2)
                amount = int(m.group(3))
                
                horses_list = [h.strip() for h in horses_raw.split(',')]
                
                bet_type = '3連複'
                method = 'BOX' if method_raw == 'BOX' else method_raw
                horses_obj = horses_list
                
                bets.append({
                    'type': bet_type,
                    'method': method,
                    'horses': horses_obj,
                    'total_amount': amount,
                    'raw_line': line
                })
        else:
            # 既存モデル (統合レポートの既存セクション)
            existing_sec = re.search(r'### 【既存モデルの推奨買い目】(.*?)(?:### 【CatBoost|\Z)', race_block, re.DOTALL)
            if existing_sec:
                existing_text = existing_sec.group(1)
                for line in existing_text.split('\n'):
                    line = line.strip()
                    m_bet = re.match(r'-\s+(.*?)\s+(BOX|流し|SINGLE)\s+\((.*?)\):\s+(\d+)円\s+\((\d+)点\)', line)
                    if m_bet:
                        kind = m_bet.group(1)
                        if '3連複' in kind:
                            method = m_bet.group(2)
                            horses_str = m_bet.group(3)
                            amount = int(m_bet.group(4))
                            
                            if method == 'BOX':
                                clean_str = horses_str.replace('BOX', '').strip()
                                nums = re.findall(r'\d+', clean_str)
                                if nums:
                                    bets.append({
                                        'type': '3連複',
                                        'method': 'BOX',
                                        'horses': nums,
                                        'total_amount': amount,
                                        'raw_line': line
                                    })
                            
                            # Note: 他の方式（流しなど）の実装が必要な場合はここに追加する

        if bets:
            bets_by_race[race_id] = bets

    return bets_by_race

def calculate_combinations(bet_type, method, horses):
    if method == 'BOX':
        n = len(horses)
        if bet_type == '3連複':
            return math.comb(n, 3)
        if bet_type in ['馬連', 'ワイド']:
            return math.comb(n, 2)
        if bet_type in ['単勝', '複勝']:
            return n
    elif method == '流し':
        axis_count = len(horses.get('axis', []))
        opp_count = len(horses.get('opponents', []))
        if bet_type == '3連複':
            if axis_count == 1:
                return math.comb(opp_count, 2)
            if axis_count == 2:
                return opp_count
        if bet_type in ['馬連', 'ワイド']:
            return opp_count
    return 1

def login_netkeiba(driver, secrets):
    logger.info('[LOGIN] ログイン中...')
    driver.get('https://regist.netkeiba.com/account/?pid=login')
    wait = WebDriverWait(driver, 10)
    try:
        user_input = wait.until(EC.presence_of_element_located((By.NAME, 'login_id')))
        pass_input = driver.find_element(By.NAME, 'pswd')
        user_input.send_keys(secrets['email'])
        pass_input.send_keys(secrets['password'])
        try:
            btn = driver.find_element(By.XPATH, "//input[@type='image' and contains(@alt, 'ログイン')]")
        except Exception:
            btn = driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
        btn.click()
        wait.until(EC.url_contains('netkeiba.com'))
        logger.info('[LOGIN] 成功')
    except Exception as e:
        logger.error(f'[LOGIN] 失敗: {e}')
        save_evidence(driver, 'login', 'error')
        sys.exit(1)

def input_marks(driver, race_id, bets):
    try:
        if not bets:
            return
        first_bet = bets[0]
        if first_bet['method'] == '流し':
            target_horse = first_bet['horses']['axis'][0]
        else:
            target_horse = first_bet['horses'][0]

        horse_str = str(int(target_horse))
        logger.info(f'[{race_id}] ◎ → 馬番 {horse_str}')

        xpath = f"//tr[contains(@class, 'HorseList') and td[text()='{horse_str}']]"
        row = driver.find_element(By.XPATH, xpath)
        row_id = row.get_attribute('id')
        internal_id = row_id.replace('tr_', '')
        mark_id = f'm1-{internal_id}'
        label_id = f'ml1-{internal_id}'

        input_el = driver.find_element(By.ID, mark_id)
        if not input_el.is_selected():
            label_btn = driver.find_element(By.ID, label_id)
            driver.execute_script('arguments[0].click();', label_btn)
            time.sleep(0.5)
    except Exception as e:
        logger.warning(f'[{race_id}] 印入力エラー (スキップ): {e}')

def ensure_shutuba_page(driver, race_id):
    url = f'https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}'
    logger.info(f'[{race_id}] Navigate: {url}')
    driver.get(url)
    time.sleep(4)
    handle_popups(driver)
    try:
        driver.find_element(By.CLASS_NAME, 'Vote')
        logger.info(f'[{race_id}] 出馬表ページ読み込み成功')
        return True
    except Exception:
        logger.error(f'[{race_id}] 出馬表ページ読み込み失敗')
        save_evidence(driver, race_id, 'shutuba_fail')
        return False

def perform_betting(driver, race_id, bets):
    input_marks(driver, race_id, bets)
    handle_popups(driver)
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

    if not input_btn:
        logger.error(f'[{race_id}] 「買い目を入力する」ボタンが見つかりません')
        save_evidence(driver, race_id, 'input_btn_missing')
        return False

    driver.execute_script('arguments[0].scrollIntoView({block: "center"});', input_btn)
    time.sleep(0.5)
    try:
        input_btn.click()
    except ElementClickInterceptedException:
        handle_popups(driver)
        driver.execute_script('arguments[0].click();', input_btn)

    time.sleep(2)
    handle_popups(driver)

    try:
        WebDriverWait(driver, 10).until(lambda d: 'ipat_sp' in d.current_url or len(d.window_handles) > 1)
    except TimeoutException:
        logger.error(f'[{race_id}] IPATページへの遷移失敗')
        return False

    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[-1])

    for i, bet in enumerate(bets):
        try:
            handle_popups(driver)
            b_type = bet['type']
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.Col4 li')))

            click_success = False
            for btn in driver.find_elements(By.CSS_SELECTOR, 'ul.Col4 li'):
                if b_type in btn.text:
                    btn.click()
                    click_success = True
                    break
            if not click_success:
                logger.warning(f'[{race_id}] 券種ボタン {b_type} が見つかりません')

            target_method = '通常'
            if bet['method'] == 'BOX':
                target_method = 'ボックス'
            elif bet['method'] == '流し':
                target_method = 'ながし'

            for m in driver.find_elements(By.XPATH, "//div[contains(text(), '方式選択')]/following-sibling::ul//li"):
                if target_method in m.text:
                    m.click()
                    time.sleep(0.5)
                    break

            if bet['method'] == '流し':
                for h in bet['horses']['axis']:
                    h_val = str(int(h))
                    try:
                        driver.execute_script(f"document.getElementById('uc-0-{h_val}').click();")
                    except:
                        pass
                    time.sleep(0.1)
                for h in bet['horses']['opponents']:
                    h_val = str(int(h))
                    try:
                        driver.execute_script(f"document.getElementById('uc-1-{h_val}').click();")
                    except:
                        pass
                    time.sleep(0.1)
            else:
                for h in bet['horses']:
                    try:
                        h_val = str(int(h))
                        tr_id = f'tr_{h_val}'
                        tr = driver.find_element(By.ID, tr_id)
                        inp = tr.find_element(By.TAG_NAME, 'input')
                        lbl = tr.find_element(By.TAG_NAME, 'label')
                        if not inp.is_selected():
                            driver.execute_script('arguments[0].click();', lbl)
                    except Exception as he:
                        logger.warning(f'[{race_id}] 馬番 {h} 選択エラー: {he}')
                        save_evidence(driver, race_id, f'horse_{h}_fail')
                    time.sleep(0.1)

            combos = calculate_combinations(bet['type'], bet['method'], bet['horses'])
            unit_price = (bet['total_amount'] // combos // 100) * 100
            if unit_price < 100:
                unit_price = 100
            logger.info(f'[{race_id}] Bet {i+1}: {b_type} {bet["method"]} {bet["horses"]} combos={combos} unit={unit_price}')

            money_input = driver.find_element(By.NAME, 'money')
            money_input.clear()
            money_input.send_keys(str(unit_price // 100))

            try:
                add_btn = driver.find_element(By.XPATH, "//button[contains(text(), '追加')]")
                driver.execute_script('arguments[0].click();', add_btn)
            except Exception:
                try:
                    add_btn = driver.find_element(By.CSS_SELECTOR, 'button.Common_Btn')
                    driver.execute_script('arguments[0].click();', add_btn)
                except Exception:
                    pass

            time.sleep(1)
            if check_error_popup(driver):
                continue
        except Exception as e:
            logger.error(f'[{race_id}] Bet {i} error: {e}')

    try:
        set_btn = None
        for sel in [(By.CLASS_NAME, 'SetBtn'), (By.XPATH, "//button[contains(text(), '買い目をセットする')]")]:
            try:
                set_btn = driver.find_element(*sel)
                break
            except Exception:
                continue

        if set_btn:
            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', set_btn)
            time.sleep(0.5)
            driver.execute_script('arguments[0].click();', set_btn)
            try:
                WebDriverWait(driver, 10).until(lambda d: 'shutuba' in d.current_url)
                logger.info(f'[{race_id}] 出馬表に戻りました')
            except TimeoutException:
                logger.error(f'[{race_id}] 出馬表への遷移失敗')
                return False
        else:
            logger.error(f'[{race_id}] SetBtn が見つかりません')
            save_evidence(driver, race_id, 'setbtn_missing')
            return False
    except Exception as e:
        logger.error(f'[{race_id}] Setフェーズエラー: {e}')
        return False

    try:
        final_btn = None
        try:
            final_btn = driver.find_element(By.ID, f'act-bet_{race_id}')
        except Exception:
            for sel in [(By.ID, 'bet_button_add'), (By.CSS_SELECTOR, "button[id^='act-bet_']"), (By.CSS_SELECTOR, '.BetBtn')]:
                try:
                    final_btn = driver.find_element(*sel)
                    break
                except Exception:
                    continue

        if final_btn:
            driver.execute_script('arguments[0].scrollIntoView({block: "center"});', final_btn)
            time.sleep(0.5)
            driver.execute_script('arguments[0].click();', final_btn)
            logger.info(f'[{race_id}] 最終登録ボタンクリック')
            time.sleep(4)
            err = check_error_popup(driver)
            if err:
                logger.error(f'[{race_id}] 登録エラー: {err}')
                return False
            return True
        else:
            logger.error(f'[{race_id}] 最終登録ボタンが見つかりません')
            save_evidence(driver, race_id, 'finalvote_missing')
            return False
    except Exception as e:
        logger.error(f'[{race_id}] 登録フェーズエラー: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(description="指定された予想レポートから俺プロへ自動投票するスクリプト")
    parser.add_argument("--report", type=str, required=True, help="対象レポートファイルのパス")
    parser.add_argument("--target_type", type=str, default="catboost", choices=["catboost", "existing"], help="対象とする買い目の抽出方法 (デフォルト: catboost)")
    args = parser.parse_args()

    logger.info(f'=== 俺プロ自動登録開始 (Target: {args.target_type}) ===')
    logger.info(f'Report: {args.report}')

    bets_by_race = parse_report(args.report, args.target_type)
    if not bets_by_race:
        logger.error("買い目が見つかりませんでした。")
        return

    logger.info(f'パース済みレース数: {len(bets_by_race)}')

    secrets = load_secrets()
    driver = setup_driver()

    success_count = 0
    fail_count = 0

    try:
        login_netkeiba(driver, secrets)

        for race_id in sorted(bets_by_race.keys()):
            bets = bets_by_race[race_id]
            if not bets:
                continue
            logger.info(f'\n--- {race_id} ({len(bets)}件の買い目) ---')
            if ensure_shutuba_page(driver, race_id):
                if perform_betting(driver, race_id, bets):
                    logger.info(f'[{race_id}] ✅ SUCCESS')
                    success_count += 1
                else:
                    logger.error(f'[{race_id}] ❌ FAILED')
                    fail_count += 1
            else:
                logger.error(f'[{race_id}] ❌ ページ読み込み失敗')
                fail_count += 1
            time.sleep(1)

        logger.info(f'\n=== 完了: 成功 {success_count}R / 失敗 {fail_count}R ===')
    finally:
        logger.info('終了しました。')
        driver.quit()

if __name__ == '__main__':
    main()
