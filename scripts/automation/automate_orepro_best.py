"""
最優秀モデル/戦略の買い目を俺プロに自動登録 (v17ベースの安定版)
- 最優秀: lgbm / box4_sanrenpuku (回収率211.5%)
- 対象レポート: reports/prediction_20260221_all_models.md
- 対象日: 2026/02/21 (土)
"""

import json
import re
import time
import os
import sys
import logging
import math
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# ===== 設定 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_FILE = os.path.join(PROJECT_ROOT, 'reports', 'prediction_20260221_all_models.md')
BEST_MODEL_JSON = os.path.join(PROJECT_ROOT, 'reports', 'best_model_strategy.json')
SECRETS_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'netkeiba_secrets.json')
LOG_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'debug', 'screenshots_orepro_best')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'orepro_best_20260221.log'),
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


def load_best_model_strategy():
    if not os.path.exists(BEST_MODEL_JSON):
        return 'lgbm', 'box4_sanrenpuku'
    with open(BEST_MODEL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['model'], data['strategy']


def parse_report(target_model: str, target_strategy: str) -> dict:
    """
    新形式レポートから対象モデル/戦略の買い目をパース
    ## 東京1R (202605010701)
    ### モデル: lgbm / 戦略: box4_sanrenpuku
    - 3連複 BOX (2,5,1,13 BOX): 400円 (4点)
    """
    bets_by_race = {}
    current_race_id = None
    in_target_section = False
    target_header = f"モデル: {target_model} / 戦略: {target_strategy}"

    if not os.path.exists(REPORT_FILE):
        logger.error(f'レポートが見つかりません: {REPORT_FILE}')
        return {}

    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        # ## 東京1R (202605010701) 形式
        m_race = re.search(r'## (東京|阪神|小倉)(\d+)R\s*\((\d+)\)', line)
        if m_race:
            current_race_id = m_race.group(3)
            bets_by_race[current_race_id] = []
            in_target_section = False
            continue

        # ### モデル: lgbm / 戦略: box4_sanrenpuku
        if line.startswith('###'):
            in_target_section = target_header in line
            continue

        if in_target_section and current_race_id and line.startswith('- '):
            m = re.match(
                r'- (単勝|複勝|馬連|ワイド|馬単|3連複|3連単)\s+(BOX|流し|SINGLE|Formation)\s+\((.+?)\):\s+(\d+)円',
                line
            )
            if not m:
                continue
            bet_type = m.group(1)
            method_raw = m.group(2)
            combo_str = m.group(3)
            amount = int(m.group(4))

            if '軸:' in combo_str and '相手:' in combo_str:
                m_ax = re.search(r'軸:([\d,]+)', combo_str)
                m_op = re.search(r'相手:([\d,]+)', combo_str)
                axis = [h.strip() for h in m_ax.group(1).split(',') if h.strip().isdigit()]
                opponents = [h.strip() for h in m_op.group(1).split(',') if h.strip().isdigit()]
                horses = {'axis': axis, 'opponents': opponents}
                method = '流し'
            else:
                horse_str = re.sub(r'\s*BOX\s*', '', combo_str)
                horses_raw = re.split(r'[,\s]+', horse_str.strip())
                horses = [h.strip() for h in horses_raw if h.strip().isdigit()]
                if method_raw.upper() == 'BOX':
                    method = 'BOX'
                elif method_raw == 'SINGLE':
                    method = 'SINGLE'
                    horses = horses[:1]
                else:
                    method = method_raw

            bets_by_race[current_race_id].append({
                'type': bet_type, 'method': method, 'horses': horses,
                'total_amount': amount, 'raw_line': line.strip()
            })

    return {k: v for k, v in bets_by_race.items() if v}


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
    """出馬表に◎印を入力"""
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
    # Step 0: 印入力
    input_marks(driver, race_id, bets)

    # Step 1: 「買い目を入力する」ボタン
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

    # Step 2: 買い目を入力
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

            # 方式選択
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

            # 馬番選択
            if bet['method'] == '流し':
                for h in bet['horses']['axis']:
                    h_val = str(int(h))
                    try:
                        driver.execute_script(f"document.getElementById('uc-0-{h_val}').click();")
                    except Exception as he:
                        logger.warning(f'軸馬 {h_val} 選択エラー: {he}')
                    time.sleep(0.1)
                for h in bet['horses']['opponents']:
                    h_val = str(int(h))
                    try:
                        driver.execute_script(f"document.getElementById('uc-1-{h_val}').click();")
                    except Exception as he:
                        logger.warning(f'相手馬 {h_val} 選択エラー: {he}')
                    time.sleep(0.1)
            else:
                # BOX方式: tr → input → label を経由してclick (v14/v17と同じ方法)
                for h in bet['horses']:
                    try:
                        h_val = str(int(h))
                        tr_id = f'tr_{h_val}'
                        logger.info(f'馬番 {h_val} 選択中 ({tr_id})')
                        tr = driver.find_element(By.ID, tr_id)
                        inp = tr.find_element(By.TAG_NAME, 'input')
                        lbl = tr.find_element(By.TAG_NAME, 'label')
                        if not inp.is_selected():
                            driver.execute_script('arguments[0].click();', lbl)
                    except Exception as he:
                        logger.warning(f'[{race_id}] 馬番 {h} 選択エラー: {he}')
                        save_evidence(driver, race_id, f'horse_{h}_fail')
                    time.sleep(0.1)

            # 金額
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

    # Step 3: SetBtn
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

    # Step 4: 最終登録
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
    best_model, best_strategy = load_best_model_strategy()
    logger.info(f'=== 最優秀モデル/戦略: {best_model} / {best_strategy} ===')

    bets_by_race = parse_report(best_model, best_strategy)
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
            logger.info(f'\n--- {race_id} ({len(bets)}件) ---')
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
        logger.info('ブラウザを開いたまま終了します（確認用）。手動でご確認ください。')
        # driver.quit()


if __name__ == '__main__':
    main()
