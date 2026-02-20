"""
最優秀モデル/戦略の買い目を俺プロ（orePro）に自動登録するスクリプト
- 対象レポート: reports/prediction_20260221_all_models.md
- 対象モデル/戦略: best_model_strategy.json から読み込み
- SeleniumでChrome操作 (automate_orepro_v17.py ベース)
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

# レース日付: 2026/02/21
RACE_DATE = '20260221'

# 会場コード: JRA場コード
VENUE_MAP = {
    '東京': '05',
    '阪神': '09',
    '小倉': '10',
}

# 各会場の開催回と日目  (前回調査結果)
KAI_DAY_MAP = {
    '東京': ('01', '07'),    # 第1回 7日目
    '阪神': ('01', '01'),    # 第1回 1日目
    '小倉': ('01', '07'),    # 第1回 7日目
}

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f'orepro_best_{RACE_DATE}.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger()
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logger.addHandler(console)


def build_race_id(venue_name: str, race_num: int) -> str:
    """会場名と出走番号からrace_idを生成"""
    jyo = VENUE_MAP.get(venue_name, '05')
    kai, day = KAI_DAY_MAP.get(venue_name, ('01', '01'))
    year = RACE_DATE[:4]
    return f"{year}{jyo}{kai}{day}{race_num:02d}"


def load_best_model_strategy():
    """最優秀モデル/戦略をJSONから読み込む"""
    if not os.path.exists(BEST_MODEL_JSON):
        logger.warning("best_model_strategy.json が見つかりません。デフォルト (lgbm/box4_sanrenpuku) を使用します。")
        return 'lgbm', 'box4_sanrenpuku'
    with open(BEST_MODEL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['model'], data['strategy']


def parse_report(target_model: str, target_strategy: str) -> dict:
    """
    レポートから対象モデル/戦略の買い目をパースする
    Returns: {race_id: [bets...]}
    """
    bets_by_race = {}
    current_race_id = None
    in_target_section = False

    if not os.path.exists(REPORT_FILE):
        logger.error(f"レポートファイルが見つかりません: {REPORT_FILE}")
        return {}

    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 表示用文字列を組み立て
    target_header = f"モデル: {target_model} / 戦略: {target_strategy}"

    for line in lines:
        line = line.rstrip()

        # ## 東京1R (202605010701) 形式
        m_race = re.search(r'## (東京|阪神|小倉)(\d+)R\s*\((\d+)\)', line)
        if m_race:
            current_race_id = m_race.group(3)  # race_idをそのまま使う
            bets_by_race[current_race_id] = []
            in_target_section = False
            continue

        # ### モデル: lgbm / 戦略: box4_sanrenpuku
        if line.startswith('###') and target_header in line:
            in_target_section = True
            continue
        elif line.startswith('###'):
            in_target_section = False
            continue

        # 買い目行のパース: - 3連複 BOX (2,5,1,13 BOX): 400円 (4点)
        if in_target_section and current_race_id and line.startswith('- '):
            try:
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

                # 馬番のパース
                if '軸:' in combo_str and '相手:' in combo_str:
                    m_axis = re.search(r'軸:([\d,]+)', combo_str)
                    m_opp = re.search(r'相手:([\d,]+)', combo_str)
                    axis = [int(h) for h in m_axis.group(1).split(',') if h.strip().isdigit()]
                    opponents = [int(h) for h in m_opp.group(1).split(',') if h.strip().isdigit()]
                    horses = {'axis': axis, 'opponents': opponents}
                    method = '流し'
                else:
                    # BOX形式: "2,5,1,13 BOX" -> [2,5,1,13]
                    horse_str = re.sub(r'\s*BOX\s*', '', combo_str)
                    horses_raw = re.split(r'[,\s]+', horse_str.strip())
                    horses = [int(h) for h in horses_raw if h.strip().isdigit()]
                    if method_raw.upper() == 'BOX':
                        method = 'BOX'
                    elif method_raw == 'SINGLE':
                        method = 'SINGLE'
                        horses = horses[:1]  # 単勝は1頭
                    else:
                        method = method_raw

                bets_by_race[current_race_id].append({
                    'type': bet_type,
                    'method': method,
                    'horses': horses,
                    'total_amount': amount,
                    'raw_line': line.strip()
                })
            except Exception as e:
                logger.warning(f"行のパースエラー '{line.strip()}': {e}")

    # 買い目がないレースは除外
    bets_by_race = {k: v for k, v in bets_by_race.items() if v}
    return bets_by_race


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


def handle_popups(driver):
    try:
        WebDriverWait(driver, 0.5).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
    except TimeoutException:
        pass
    try:
        for btn in driver.find_elements(By.XPATH, "//button[contains(text(), 'はい')]"):
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
                return
    except Exception:
        pass
    try:
        for btn in driver.find_elements(By.CSS_SELECTOR, '.swal-button--confirm'):
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
    except Exception:
        pass


def login_netkeiba(driver, secrets):
    logger.info('[LOGIN] ログインページへアクセス...')
    driver.get('https://regist.netkeiba.com/account/?pid=login')
    wait = WebDriverWait(driver, 10)
    try:
        user_input = wait.until(EC.presence_of_element_located((By.NAME, 'login_id')))
        pass_input = driver.find_element(By.NAME, 'pswd')
        user_input.send_keys(secrets['email'])
        pass_input.send_keys(secrets['password'])
        try:
            login_btn = driver.find_element(By.XPATH, "//input[@type='image' and contains(@alt, 'ログイン')]")
        except Exception:
            login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'ログイン')]")
        login_btn.click()
        wait.until(EC.url_contains('netkeiba.com'))
        logger.info('[LOGIN] 成功')
    except Exception as e:
        logger.error(f'[LOGIN] 失敗: {e}')
        sys.exit(1)


def ensure_shutuba_page(driver, race_id: str) -> bool:
    url = f'https://orepro.netkeiba.com/bet/shutuba.html?mode=init&race_id={race_id}'
    logger.info(f'[{race_id}] Navigate to: {url}')
    driver.get(url)
    time.sleep(4)
    handle_popups(driver)
    try:
        driver.find_element(By.CLASS_NAME, 'Vote')
        logger.info(f'[{race_id}] 出馬表ページ読み込み成功')
        return True
    except Exception:
        logger.error(f'[{race_id}] 出馬表ページの読み込みに失敗')
        return False


def calculate_combinations(bet_type: str, method: str, horses) -> int:
    if method == 'BOX':
        n = len(horses) if isinstance(horses, list) else 0
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
    elif method == 'SINGLE':
        return 1
    return 1


def perform_betting(driver, race_id: str, bets: list) -> bool:
    handle_popups(driver)

    # 買い目を入力するボタンを探す
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

            for btn in driver.find_elements(By.CSS_SELECTOR, 'ul.Col4 li'):
                if b_type in btn.text:
                    btn.click()
                    break

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
                    driver.execute_script(f"document.getElementById('uc-0-{h}').click();")
                    time.sleep(0.1)
                for h in bet['horses']['opponents']:
                    driver.execute_script(f"document.getElementById('uc-1-{h}').click();")
                    time.sleep(0.1)
            else:
                for h in (bet['horses'] if isinstance(bet['horses'], list) else []):
                    driver.execute_script(f"document.getElementById('tr_{h}').click();")
                    time.sleep(0.1)

            combos = calculate_combinations(bet['type'], bet['method'], bet['horses'])
            unit_price = (bet['total_amount'] // combos // 100) * 100
            if unit_price < 100:
                unit_price = 100

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
            logger.info(f'[{race_id}] Bet {i+1}: {b_type} {bet["method"]} {bet["horses"]} {bet["total_amount"]}円')
        except Exception as e:
            logger.error(f'[{race_id}] Bet {i} エラー: {e}')

    # セットボタン
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
            logger.error(f'[{race_id}] SetBtnが見つかりません')
            return False
    except Exception as e:
        logger.error(f'[{race_id}] セットフェーズエラー: {e}')
        return False

    # 最終登録ボタン
    try:
        final_btn = None
        for sel in [
            (By.ID, f'act-bet_{race_id}'),
            (By.ID, 'bet_button_add'),
            (By.CSS_SELECTOR, "button[id^='act-bet_']"),
            (By.CSS_SELECTOR, '.BetBtn')
        ]:
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
            return True
        else:
            logger.error(f'[{race_id}] 最終登録ボタンが見つかりません')
            return False
    except Exception as e:
        logger.error(f'[{race_id}] 登録フェーズエラー: {e}')
        return False


def main():
    best_model, best_strategy = load_best_model_strategy()
    logger.info(f"=== 最優秀モデル/戦略: {best_model} / {best_strategy} ===")

    bets_by_race = parse_report(best_model, best_strategy)
    if not bets_by_race:
        logger.error(f"レポートから '{best_model}/{best_strategy}' の買い目が見つかりませんでした。")
        return

    logger.info(f"パース済みレース数: {len(bets_by_race)}")
    for rid, bets in list(bets_by_race.items())[:3]:
        logger.info(f"  {rid}: {len(bets)}件 -> {[b['raw_line'] for b in bets]}")

    # ドライバー起動
    secrets = load_secrets()
    driver = setup_driver()

    try:
        login_netkeiba(driver, secrets)

        success_count = 0
        fail_count = 0

        for race_id, bets in sorted(bets_by_race.items()):
            if not bets:
                continue
            logger.info(f'\n--- {race_id} ---')
            if ensure_shutuba_page(driver, race_id):
                if perform_betting(driver, race_id, bets):
                    logger.info(f'[{race_id}] SUCCESS')
                    success_count += 1
                else:
                    logger.error(f'[{race_id}] FAILED')
                    fail_count += 1
            else:
                logger.error(f'[{race_id}] ページ読み込みに失敗')
                fail_count += 1
            time.sleep(1)

        logger.info(f'\n=== 完了: 成功 {success_count}R / 失敗 {fail_count}R ===')
    finally:
        logger.info('ブラウザを開いたまま終了します（確認用）')
        # driver.quit()


if __name__ == '__main__':
    main()
