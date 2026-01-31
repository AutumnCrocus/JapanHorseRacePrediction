"""
IPAT直接連携モジュール（Selenium版 - Smartphone Site）
JRA IPAT (スマートフォン版) にアクセスして投票画面を自動操作するモジュール
Reference: https://zenn.dev/_lambda314/articles/e4ceaa81b045c5
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.alert import Alert
import time
from typing import List, Dict, Any, Optional
import os
import datetime
import traceback
import sys

class IpatDirectAutomator:
    """IPAT直接連携クラス（Selenium版 - Smartphone Site）"""
    
    # 定数
    JRA_IPAT_URL = "https://www.ipat.jra.go.jp/sp/"
    WAIT_SEC = 2
    
    # 曜日リスト (記事準拠)
    DOW_LST = ["月", "火", "水", "木", "金", "土", "日"]
    # レース会場リスト (記事準拠)
    PLACE_LST = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    
    def __init__(self):
        """初期化"""
        self.driver = None
        self.wait_timeout = 10
        
    def _save_debug_screenshot(self, driver, name: str):
        """デバッグ用スクリーンショットを保存"""
        try:
            screenshot_dir = os.path.join(os.getcwd(), 'debug_screenshots')
            os.makedirs(screenshot_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(screenshot_dir, f'{name}_{timestamp}.png')
            driver.save_screenshot(filepath)
            print(f"Screenshot saved: {filepath}")
        except Exception as e:
            print(f"Failed to save screenshot: {e}")

    def _judge_day_of_week(self, date_nm: str) -> str:
        """日付文字列(YYYYMMDD)から曜日文字を取得"""
        try:
            date_dt = datetime.datetime.strptime(str(date_nm), "%Y%m%d")
            # isoweekday: 月曜=1 ... 日曜=7
            nm = date_dt.isoweekday()
            return self.DOW_LST[nm - 1]
        except ValueError:
            return ""

    def _click_css_selector(self, selector: str, index: int = 0):
        """指定したCSSセレクタの要素をクリックする"""
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            if len(elements) > index:
                element = elements[index]
                # 記事同様、JavaScriptでのクリックも併用検討だが、まずは標準クリック
                # self.driver.execute_script("arguments[0].click();", element) 
                element.click()
                time.sleep(self.WAIT_SEC)
                return True
            else:
                print(f"Warning: Element not found or index out of range: {selector}[{index}]")
                return False
        except Exception as e:
            print(f"Click Error ({selector}): {e}")
            return False

    def _save_snapshot(self, name):
        """Save screenshot and page source for debugging."""
        if not self.driver: return
        
        try:
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            filename_base = f"debug_{timestamp}_{name}"
            
            # Screenshot
            self.driver.save_screenshot(f"{filename_base}.png")
            
            # Page Source
            with open(f"{filename_base}.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
                
            print(f"[Snapshot] Saved {filename_base} (.png, .html)")
            
            # Print current active page ID
            try:
                active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                print(f"[Snapshot] Active Page ID: {active.get_attribute('id')}")
            except:
                print("[Snapshot] No active page found.")
                
        except Exception as e:
            print(f"Failed to save snapshot {name}: {e}")

    def login(self, inetid: str, subscriber_no: str, pin: str, pars_no: str) -> tuple[bool, str]:
        """
        IPATログイン画面で認証を実行（スマホ版）
        
        Args:
            inetid: INET-ID
            subscriber_no: 加入者番号
            pin: 暗証番号
            pars_no: P-ARS番号
        """
        try:
            # Chromeオプション設定
            options = Options()
            # options.add_argument("-headless") # デバッグ時はHeadless無効
            options.add_argument('--start-maximized')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            # ドライバ起動
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            print(f"IPAT(SP)へアクセス中: {self.JRA_IPAT_URL}")
            self.driver.get(self.JRA_IPAT_URL)
            time.sleep(self.WAIT_SEC)
            
            # Helper: 安全な入力関数
            def safe_send_keys(element, value, name="unknown"):
                try:
                    print(f"Inputting to {name}...")
                    # まずクリックしてフォーカス
                    try:
                        element.click()
                    except:
                        pass
                    
                    # 標準的なclear/send_keysを試す
                    element.clear()
                    element.send_keys(value)
                except Exception as e:
                    print(f"Standard input failed for {name}: {e}. Trying JS...")
                    # 失敗したらJSで直接値をセット
                    self.driver.execute_script("arguments[0].value = arguments[1];", element, value)
                    # イベント発火 (input, change, blur)
                    # JQuery Mobileなどでは input/change イベントでモデル更新することが多い
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", element)
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", element)
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('blur', { bubbles: true }));", element)

            # 1. INET-ID入力画面かどうかのチェック
            # 別画面、もしくは同じ画面の別stateの可能性がある
            try:
                # 入力欄があるか (id="inetid" または name="inetid" で visibleなもの)
                inetid_inputs = self.driver.find_elements(By.CSS_SELECTOR, "#inetid, input[name='inetid']")
                visible_inetid = [el for el in inetid_inputs if el.is_displayed()]
                
                if visible_inetid:
                    print("INET-ID入力画面を検出")
                    if inetid:
                        safe_send_keys(visible_inetid[0], inetid, "inetid")
                        # ログイン/次へボタン (汎用的)
                        if not self._click_css_selector("a[onclick^='javascript']", 0):
                            self._click_css_selector("a", 0)
                        time.sleep(self.WAIT_SEC)
                    else:
                        print("Warning: INET-ID screen detected but no INET-ID provided.")
                else:
                    print("INET-ID入力フィールド（表示）なし。加入者情報入力へ進みます。")
            except Exception as e:
                print(f"INET-ID check warning: {e}")

            # 2. 加入者情報入力画面
            # JQuery Mobile Source:
            # id="userid" (加入者番号)
            # id="password" (暗証番号)
            # id="pars" (P-ARS番号)
            
            try:
                time.sleep(1)
                
                # 加入者番号 (id="userid")
                try:
                    user_inputs = self.driver.find_elements(By.ID, "userid")
                    visible_inputs = [el for el in user_inputs if el.is_displayed()]
                    
                    if visible_inputs:
                        safe_send_keys(visible_inputs[0], subscriber_no, "subscriber_no")
                    else:
                        # まだ画面遷移していない？あるいは hidden inputsしかない？
                        print("Subscriber input 'userid' not found (visible). Checking raw source dump if stuck.")
                        # 念のため name='i' の visible も探すが、JQM版では id="userid" が正
                        raise Exception("Subscriber input 'userid' not found")
                        
                except Exception as e:
                    print(f"Error processing 'userid': {e}")
                    raise e

                # 暗証番号 (id="password")
                try:
                    pass_inputs = self.driver.find_elements(By.ID, "password")
                    visible_pass = [el for el in pass_inputs if el.is_displayed()]
                    
                    if visible_pass:
                        safe_send_keys(visible_pass[0], pin, "pin")
                    else:
                        raise Exception("PIN input 'password' not found")
                except Exception as e:
                    print(f"Error processing 'password': {e}")
                    raise e
                    
                # P-ARS番号 (id="pars")
                try:
                    pars_inputs = self.driver.find_elements(By.ID, "pars")
                    visible_pars = [el for el in pars_inputs if el.is_displayed()]
                    
                    if visible_pars:
                        safe_send_keys(visible_pars[0], pars_no, "pars_no")
                    else:
                        print("Warning: P-ARS input 'pars' not found")
                except Exception as e:
                    print(f"Error processing 'pars': {e}")

                # ログインボタン
                # <a onclick="JavaScript:ToSPMenu();return false;" class="ui-link">ログイン</a>
                btn_found = False
                
                # Text content "ログイン"
                try:
                    xpath = "//a[contains(text(), 'ログイン')]"
                    btns = self.driver.find_elements(By.XPATH, xpath)
                    for btn in btns:
                        if btn.is_displayed():
                            self.driver.execute_script("arguments[0].click();", btn)
                            btn_found = True
                            print("Login button clicked (text match)")
                            break
                except:
                    pass
                
                if not btn_found:
                    # onclick="JavaScript:ToSPMenu();"
                    try:
                        btn = self.driver.find_element(By.CSS_SELECTOR, "a[onclick*='ToSPMenu']")
                        self.driver.execute_script("arguments[0].click();", btn)
                        btn_found = True
                        print("Login button clicked (ToSPMenu)")
                    except:
                        pass
                
                if not btn_found:
                    print("Login button specific methods failed. Trying generic...")
                    # ui-btn-active or similar JQM class?
                    # Fallback to any button-like link
                    pass

            except Exception as e:
                print(f"Login input critical error: {e}")
                self._save_debug_screenshot(self.driver, "login_input_error")
                return False, f"ログイン入力エラー: {e}"

            time.sleep(self.WAIT_SEC)
            
            # 同意画面対応
            try:
                if self.driver.find_elements(By.ID, "contract_area"):
                    if self.driver.find_element(By.ID, "contract_area").is_displayed():
                        print("同意画面を検出")
                        # 全文表示
                        try:
                            self.driver.execute_script("DispAllContract();") # JS関数直接呼び出し
                            time.sleep(1)
                        except:
                            pass
                        
                        # Check if we need to click "To Amount" or "Set" to proceed from Horse Page
                        # Typically for multi-horse bets (like Umaren), there is a button "金額入力画面へ"
                        try:
                            time.sleep(1)
                            # Check if we are still on Horse/Method page
                            active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                            aid = active.get_attribute("id")
                            
                            # If we are NOT on Amount page (Amount page usually has input[name='sum'] or similar, or ID 'kin')
                            # Let's look for "金額入力画面へ" button
                            next_btns = self.driver.find_elements(By.XPATH, "//a[contains(text(), '金額入力画面へ')]")
                            for btn in next_btns:
                                if btn.is_displayed():
                                    print("Clicking 'To Amount Input' button...")
                                    self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                                    btn.click()
                                    time.sleep(2)
                                    break
                        except Exception as e:
                            print(f"Next button error: {e}")

                        # 購入枚数・金額入力
                        try:
                            print("Setting amount: 100 (Qty: 1)")
                            
                            # Wait for input
                            WebDriverWait(self.driver, 5).until(
                                lambda d: d.find_elements(By.CSS_SELECTOR, "input[type='number'], input[type='tel']")
                            )
                        except Exception as e:
                             print(f"Wait for amount input warning: {e}")

                        # 同意ボタン
                        btns = self.driver.find_elements(By.CSS_SELECTOR, ".agreeBtn")
                        if btns:
                            self.driver.execute_script("arguments[0].click();", btns[0])
                            time.sleep(self.WAIT_SEC)
            except Exception as e:
                print(f"Agreement check warning: {e}")

            # お知らせ画面スキップ
            if "announce" in self.driver.current_url or self.driver.find_elements(By.CSS_SELECTOR, "div.announce"):
                print("お知らせ画面を検出、スキップします...")
                if not self._click_css_selector("button[href^='#!/']", 0):
                    self._click_css_selector("a.button", 0)
            
            time.sleep(self.WAIT_SEC)
            
            # ログイン成功判定
            if self.driver.find_elements(By.CSS_SELECTOR, "button[href^='#!/bet/basic']") or \
               "top" in self.driver.current_url or \
               "メニュー" in self.driver.page_source or \
               (self.driver.title and "ネット投票メニュー" in self.driver.title):
                print("IPAT Login Successful")
                return True, "ログイン成功"
            else:
                self._save_debug_screenshot(self.driver, "login_failed")
                return False, "ログインに失敗しました（メニュー画面未到達）"

        except Exception as e:
            print(f"System Error: {e}")
            traceback.print_exc()
            return False, f"システムエラー: {e}"

    def vote(self, race_id: str, bets: list[dict], stop_at_confirmation: bool = False) -> tuple[bool, str]:
        """
        指定レースに投票を実行する（拡張版）
        bets: [{ 'type': '単勝', 'horses': [1], 'amount': 100, 'method': '通常'|'ボックス' }, ...]
        stop_at_confirmation: Trueの場合、合計金額入力画面で停止し、投票ボタンを押さずに終了する
        """
        if not self.driver:
            return False, "ドライバが初期化されていません"
            
        try:
            # 1. 通常投票メニューへ遷移
            print("通常投票メニューへ遷移...")
            
            # JQuery Mobile: <a class="ico_regular ui-link">通常投票</a>
            # Text match or Class match
            nav_success = False
            try:
                # Classで探す
                btn = self.driver.find_element(By.CLASS_NAME, "ico_regular")
                if btn.is_displayed():
                    btn.click()
                    nav_success = True
                else:
                    # Textで探す
                    xpath = "//a[contains(text(), '通常投票')]"
                    self.driver.find_element(By.XPATH, xpath).click()
                    nav_success = True
            except Exception as e:
                print(f"Navigation to Normal Vote failed: {e}")
            
            if not nav_success:
                 return False, "通常投票メニューへの遷移失敗"
            
            time.sleep(self.WAIT_SEC)

            # Check for Warning Page (Deposit Instruction)
            # id="warning" class="ui-page-active"
            try:
                warning_page = self.driver.find_elements(By.CSS_SELECTOR, "div#warning.ui-page-active")
                if warning_page:
                    print("Warning Page (Deposit Info) detected.")
                    go_vote_btn = self.driver.find_elements(By.ID, "GoVote")
                    if go_vote_btn and go_vote_btn[0].is_displayed():
                        print("Clicking 'GoVote' to proceed...")
                        go_vote_btn[0].click()
                        time.sleep(self.WAIT_SEC)
            except Exception as e:
                print(f"Warning page check failed: {e}")

            time.sleep(1)

            # Wait for #voteRace OR #jyo page to be active
            print("Waiting for Place Selection page (#voteRace or #jyo)...")
            active_page_id = None
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "#voteRace.ui-page-active, #jyo.ui-page-active")
                )
                print("Place Selection page is active.")
                
                # Identify which one is active
                if self.driver.find_elements(By.CSS_SELECTOR, "#voteRace.ui-page-active"):
                    active_page_id = "voteRace"
                else:
                    active_page_id = "jyo"
                print(f"Active Page ID: {active_page_id}")
                
            except Exception as e:
                print(f"Wait for Place Selection page failed: {e}")
                # Log current active page
                try:
                    active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                    print(f"Current Active Page (Fallback): {active.get_attribute('id')}")
                    active_page_id = active.get_attribute('id')
                except:
                    pass

            # 2. レース会場を選択
            # 2. レース会場を選択
            # race_id: YYYYMMDDKKRR
            jra_place_map = {
                "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", 
                "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
            }
            place_code = race_id[8:10]
            target_place_name = jra_place_map.get(place_code, "")
            
            # 日付から曜日を判定して補強 (例: "東京" -> "東京(土)")
            # race_id: 20240101...
            try:
                date_str = race_id[0:8]
                dt = datetime.datetime.strptime(date_str, "%Y%m%d")
                w_list = ["(月)", "(火)", "(水)", "(木)", "(金)", "(土)", "(日)"]
                target_dow = w_list[dt.weekday()]
            except:
                target_dow = ""
                
            print(f"Target Race: {target_place_name} {target_dow} (Code: {place_code})")
            
            place_found = False
            
            def attempt_click_place(place_name, dow):
                # #voteRace uses ul.raceInfoList, #jyo uses ul.selectList
                # Combine selectors
                css_selectors = ["ul.selectList li a", "ul.raceInfoList li a"]
                
                for selector in css_selectors:
                    links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if not links: continue
                    
                    print(f"Checking selector {selector}: {len(links)} links found.")
                    for link in links:
                        txt = link.text.strip()
                        # 空文字ならスキップ (非表示要素など)
                        if not txt: continue
                        
                        # Match Logic: Place Name MUST match. DOW SHOULD match if present in text.
                        # "東京(土)" vs "東京" + "(土)"
                        if place_name in txt:
                            # If text contains a DOW like (土), verify it matches target_dow
                            # If text is just "東京", assume it's correct?
                            
                            # Simple strict match: if target_dow is known, check if it's in text
                            if dow and dow not in txt:
                                print(f"Skipping {txt} (Matches place but not DOW {dow})")
                                continue
                            
                            print(f"Found Place Element: {txt}")
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", link)
                            try:
                                link.click()
                            except:
                                self.driver.execute_script("arguments[0].click();", link)
                            return True
                return False

            # 実行
            if attempt_click_place(target_place_name, target_dow):
                place_found = True
                time.sleep(self.WAIT_SEC)
            
            # タブ切り替え (#voteRace の場合のみ有効だが、汎用的に残すか、#jyoなら不要)
            # #jyo does not seem to have tabs. #voteRace does.
            # Only try tabs if we failed and we see tabs.
            if not place_found:
                tabs = self.driver.find_elements(By.CSS_SELECTOR, "ul.tabNav li a")
                if tabs:
                    print("Place not found in current view. Trying tabs (if any)...")
                    for tab in tabs:
                        if "selected" not in tab.get_attribute("class"):
                            print(f"Switching to tab: {tab.text}")
                            try:
                                tab.click()
                            except:
                                self.driver.execute_script("arguments[0].click();", tab)
                            time.sleep(1)
                            
                            if attempt_click_place(target_place_name, target_dow):
                                place_found = True
                                time.sleep(self.WAIT_SEC)
                                break

            if not place_found:
                 print(f"Failed to find place: {target_place_name} {target_dow}")
                 with open("debug_place_fail_source.html", "w", encoding="utf-8") as f:
                     f.write(self.driver.page_source)
                 return False, f"開催場ボタンが見つかりません: {target_place_name} {target_dow}"

            # 3. レース番号を選択
            # Wait for #race page?
            # Based on TMPL, id='race' is likely
            print("Waiting for Race Selection page (#race)...")
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "#race.ui-page-active") or \
                              d.find_elements(By.CSS_SELECTOR, "#voteRaceTable.ui-page-active")
                )
                print("Race Selection page is active.")
                try:
                    active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                    print(f"Active Page ID for Race: {active.get_attribute('id')}")
                except: pass
            except:
                print("Wait for #race timed out. Proceeding anyway.")
            
            self._save_snapshot("after_place_selection_before_race")

            # 2. レース選択
            race_num_str = str(int(race_id[10:12])) # "01" -> "1"
            target_race_text = f"{race_num_str}R"
            print(f"レース選択: {target_race_text}")
            
            race_found = False
            
            # Strict Selector logic
            race_nums = self.driver.find_elements(By.CSS_SELECTOR, "a .raceNum")
            print(f"Found {len(race_nums)} race number spans.")
            
            for rn in race_nums:
                # Exact match
                if rn.text.strip() == target_race_text:
                    link = rn.find_element(By.XPATH, "./..") # parent <a>
                    print(f"Found Race Element (Strict): {rn.text}")
                    print(f"DEBUG: Link Classes: {link.get_attribute('class')}")
                    print(f"DEBUG: Link Href: {link.get_attribute('href')}")
                    
                    # Capture current page ID before click
                    old_id = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id")
                    
                    # Click Logic
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", link)
                    time.sleep(0.5)
                    try:
                        print("Attempting Native Click...")
                        link.click()
                    except Exception as e:
                        print(f"Native Click failed: {e}. Trying JS Click...")
                        self.driver.execute_script("arguments[0].click();", link)
                        
                    race_found = True
                    
                    # Wait for Page Transition
                    print("Waiting for page transition after race click...")
                    try:
                        WebDriverWait(self.driver, 5).until(
                            lambda d: d.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id") != old_id
                        )
                        new_id = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id")
                        print(f"Page transitioned from #{old_id} to #{new_id}")
                    except:
                        print("Page transition timed out (ID did not change).")
                    
                    break
            
            if not race_found:
                 print(f"Error: Race {target_race_text} not found.")
                 self._save_snapshot("race_not_found")
                 return False, "指定されたレースが見つかりません"

            # Check where we are
            self._save_snapshot("after_race_selection_attempt")
            
            # Wait for Bet Type Selection Page (#siki) specifically
            print("Waiting for Bet Type Selection page (#siki)...")
            try:
                WebDriverWait(self.driver, 5).until(
                    lambda d: d.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id") == "siki"
                )
                print("Bet Type Selection page (#siki) is active.")
            except:
                print("Wait for #siki timed out.")

            # 3. 投票ループ

            # 3. 投票ループ
            for i, bet in enumerate(bets):
                print(f"Processing Bet: {bet['type']} {bet.get('horses')}")
                
                # Check Page State
                self._save_snapshot(f"start_loop_{i}")

                # Ensure we are on Type Selection (#siki) or able to select type
                # If we are on Horse Page (from previous 'Continue'?), go back
                try:
                    time.sleep(1)
                    active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                    aid = active.get_attribute("id")
                    print(f"Active Page at start of bet loop: {aid}")
                    
                    if aid.startswith("uma") or aid.startswith("waku"):
                       # We are on Horse selection, need to go back to Bet Type
                       print("Currently on Horse page, returning to Bet Type...")
                       headers = active.find_elements(By.CSS_SELECTOR, "header .headerNavLeftArrow a")
                       if headers:
                           # Ensure visible / interactable
                           headers[0].click()
                           time.sleep(1)
                           self._save_snapshot(f"after_back_click_{i}")
                except: pass

                # 4. 式別選択 (Bet Type)
                try:
                    # Wait for list
                    status_text = "Wait for Bet Type Buttons"
                    WebDriverWait(self.driver, 5).until(
                         lambda d: d.find_elements(By.CSS_SELECTOR, ".ui-page-active ul.selectList li a")
                    )
                    
                    btype_text = bet['type']
                    found_type = False
                    
                    # Scope to active page
                    active_page = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                    types = active_page.find_elements(By.CSS_SELECTOR, "ul.selectList li a")
                    
                    for t in types:
                        if not t.is_displayed(): continue
                        clean_text = t.text.strip().split("\n")[0]
                        if btype_text in t.text:
                            print(f"Found Bet Type button: {btype_text}")
                            try:
                                t.click()
                            except:
                                self.driver.execute_script("arguments[0].click();", t)
                            found_type = True
                            break
                    
                    if not found_type:
                        print(f"Warning: Bet Type button '{btype_text}' not found.")
                        self._save_snapshot(f"bet_type_not_found_{i}")
                
                except Exception as e:
                    print(f"Bet Type selection error: {e}")
                    self._save_snapshot(f"bet_type_error_{i}")

                # Wait for Horse Selection Page (#uma... or #waku...)
                print("Waiting for Horse Selection page...")
                try:
                    WebDriverWait(self.driver, 5).until(
                        lambda d: d.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id").startswith("uma") or \
                                  d.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id").startswith("waku") or \
                                  d.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id") == "hou"
                    )
                    active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                    aid = active.get_attribute("id")
                    print(f"Horse Selection page is active. ID: {aid}")
                    
                    # Check for "Multi Info" (マルチについて)
                    if aid == "multi_info" or "multi_info" in self.driver.current_url:
                        print("Multi Info screen detected. Clicking OK...")
                        try:
                            ok_btn = self.driver.find_element(By.CSS_SELECTOR, "a.ui-btn") # Usually OK is the main button
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", ok_btn)
                            self.driver.execute_script("arguments[0].click();", ok_btn)
                            time.sleep(1)
                            # Update active
                            active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                            aid = active.get_attribute("id")
                            print(f"After Multi Info: {aid}")
                        except Exception as e:
                            print(f"Failed to close Multi Info: {e}")

                    # If we are on Method Selection (#hou), select Method
                    if aid == "hou":
                        method = bet.get('method', '通常') # Default Normal
                        print(f"On Method Selection (#hou). Selecting '{method}'...")
                        
                        target_text = method
                        if method == 'Box' or method == 'box' or method == 'ボックス': target_text = "ボックス"
                        if method == 'Normal' or method == 'normal' or method == '通常': target_text = "通常"
                        
                        try:
                            # Search for button by text
                            method_btn = None
                            btns = self.driver.find_elements(By.CSS_SELECTOR, "ul.selectList li a")
                            for b in btns:
                                if target_text in b.text:
                                    method_btn = b
                                    break
                            
                            if method_btn:
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", method_btn)
                                method_btn.click()
                            else:
                                print(f"Method button '{target_text}' not found. Clicking first available (Fall back).")
                                btns[0].click()

                            time.sleep(1)
                            # Update active page
                            active = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                            aid = active.get_attribute("id")
                            print(f"Moved to: {aid}")
                        except Exception as e:
                            print(f"Failed to select vote method: {e}")
                            
                except:
                    print("Wait for Horse Selection page timed out.")
                
                time.sleep(1)

                # 馬番選択 (#uma... / #waku...)
                for horse in bet.get('horses', []):
                    try:
                        h_code = str(int(horse)) # "1"
                        
                        # Use simple robust selector: a[data-value='1']
                        # Backup: label for checkboxes
                        selector = f"a[data-value='{h_code}']"
                        
                        try:
                             # Find Element WITHIN Active Page
                             active_page = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active")
                             els = active_page.find_elements(By.CSS_SELECTOR, selector)
                             
                             clicked = False
                             for el in els:
                                 if el.is_displayed():
                                     print(f"Selecting Horse {h_code}...")
                                     try:
                                         el.click()
                                     except:
                                         self.driver.execute_script("arguments[0].click();", el)
                                     clicked = True
                                     break
                             
                             if not clicked:
                                 print(f"Horse element {selector} found but not displayed.")
                                 
                        except Exception as e:
                             print(f"Primary horse selector failed: {e}. Trying backups...")
                             pass
                        
                    except Exception as e:
                         print(f"Horse number {horse} selection failed: {e}")
                
                time.sleep(1)

                # Check if we need to click "To Amount" to proceed from Horse Page
                # Typically for multi-horse bets (like Umaren), there is a button "金額入力画面へ"
                try:
                    # Check if we are still on Horse/Method page (and not Amount page)
                    # Amount page usually has input fields. Horse page might not?
                    # Or check for the button explicitly.
                    next_btns = self.driver.find_elements(By.XPATH, "//a[contains(text(), '金額入力画面へ')]")
                    for btn in next_btns:
                        if btn.is_displayed():
                            print("Clicking 'To Amount Input' button...")
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                            btn.click()
                            time.sleep(2)
                            break
                except Exception as e:
                    print(f"Next button error: {e}")
                
                # 購入枚数・金額入力
                try:
                    amount = bet.get('amount', 100)
                    qty = amount // 100
                    print(f"Setting amount: {amount} (Qty: {qty})")
                    
                    # 記事: input[ng-model^='vm.nUnit'] => JQM: input type='tel' or 'number'?
                    # Only visible inputs
                    inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='number'], input[type='tel']")
                    input_set = False
                    for inp in inputs:
                        if inp.is_displayed():
                             inp.clear()
                             inp.send_keys(str(qty))
                             input_set = True
                             break
                    
                    if not input_set:
                        print("Warning: Could not find amount input field.")

                    # セットボタン (Set/Add)
                    # Look for button "セット" or "追加"
                    # Prioritize exact or simple matches
                    self._save_snapshot(f"before_set_click_{i}")
                    
                    set_btns = self.driver.find_elements(By.CSS_SELECTOR, "a.ui-btn")
                    
                    # DEBUG: Print all visible buttons AND links
                    print("DEBUG: All visible links/buttons on #kin:")
                    all_links = self.driver.find_elements(By.TAG_NAME, "a")
                    for b in all_links:
                        if b.is_displayed():
                            print(f" - Text: '{b.text}', ID: {b.get_attribute('id')}, Class: {b.get_attribute('class')}")
                    
                    target_btn = None
                    
                    # 1. Exact "セット" or "追加" or "全セット"(All Set)
                    for btn in all_links: # Check ALL links, not just ui-btn
                        if not btn.is_displayed(): continue
                        txt = btn.text.strip()
                        if txt == "セット" or txt == "追加" or txt == "全セット":
                            target_btn = btn
                            break
                    
                    # 2. Contains
                    if not target_btn:
                         for btn in all_links:
                             if not btn.is_displayed(): continue
                             txt = btn.text.strip()
                             if ("セット" in txt or "追加" in txt) and "展開" not in txt:
                                 target_btn = btn
                                 break
                    
                    # 3. Fallback (allow 展開 only if nothing else)
                    if not target_btn:
                         for btn in all_links:
                             if not btn.is_displayed(): continue
                             if "展開セット" in btn.text:
                                 target_btn = btn
                                 print("Fallback: Using 展開セット")
                                 break

                    if target_btn:
                        print(f"Clicking Set button: {target_btn.text}")
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", target_btn)
                        time.sleep(0.5)
                        try:
                            print("Attempting Native Click on Set...")
                            target_btn.click()
                        except Exception as e:
                            print(f"Native Set Click failed: {e}. Using JS Click...")
                            self.driver.execute_script("arguments[0].click();", target_btn)
                    else:
                        print("Error: Set button not found. Trying ENTER key on input...")
                        inputs[0].send_keys(Keys.ENTER)
                        
                except Exception as e:
                    print(f"Amount/Set error: {e}")
                
                time.sleep(2)
                
                # End of Cycle for this bet confirmed by presence of #toui (Vote List)
                try:
                    print("Waiting for Vote List (#toui)...")
                    WebDriverWait(self.driver, 10).until(
                        lambda d: d.find_element(By.CSS_SELECTOR, "#toui.ui-page-active")
                    )
                    print("Vote List (#toui) is active.")
                    self._save_snapshot(f"at_vote_list_{i}")
                except:
                    print("Warning: Did not reach Vote List. Set might have failed.")
                    # Fallback: Try ENTER key if we are still on #kin
                    try:
                        aid = self.driver.find_element(By.CSS_SELECTOR, ".ui-page-active").get_attribute("id")
                        if aid.startswith("kin"):
                             print("Stuck on #kin. Trying ENTER key...")
                             # Re-find input
                             inp = self.driver.find_element(By.CSS_SELECTOR, "input[type='number'], input[type='tel']")
                             inp.send_keys(Keys.ENTER)
                             time.sleep(2)
                    except: pass
                    
                    self._save_snapshot(f"set_failed_stuck_{i}")

                # If there are more bets, click "Continue Input"
                if i < len(bets) - 1:
                    try:
                         print("More bets to process. Clicking 'Continue Input'...")
                         # "馬（枠）番から続けて入力" -> Usually returns to Horse/Bet Type
                         cont_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), '続けて入力')]")
                         cont_btn.click()
                         # Wait for transition away from #toui
                         WebDriverWait(self.driver, 10).until_not(
                            lambda d: d.find_element(By.CSS_SELECTOR, "#toui.ui-page-active")
                         )
                         time.sleep(2) # Extra stability wait
                    except Exception as e:
                        print(f"Failed to click Continue: {e}")

                # End of betting loop
            
            # 5. 購入完了 (Input Finish)
            print("All bets processed. (Should be on Vote List)")
            try:
                # If we are NOT on Vote List (e.g. last bet failed), try to recover?
                # Or assume we are there.
                pass
            except: pass
            
            try:
                # Click "Input Finish" (入力終了)
                # Should be on #toui (Vote List) page
                finish_btn = WebDriverWait(self.driver, 10).until(
                    lambda d: d.find_element(By.XPATH, "//a[contains(text(), '入力終了')]")
                )
                print("Clicking 'Input Finish'...")
                finish_btn.click()
            except Exception as e:
                print(f"Failed to click Input Finish: {e}")
            
            time.sleep(2)

                # 6. 合計金額入力 (Total Amount)
            print("Waiting for Total Amount Input page...")
            try:
                # Wait for Total Amount page
                WebDriverWait(self.driver, 10).until(
                    lambda d: "合計金額入力" in d.page_source
                )
                
                # Recalculate based on bets list
                total_amount = sum(b.get('amount', 100) for b in bets)
                print(f"Inputting Total Amount: {total_amount}")
                
                # Check for input
                total_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='number'], input[type='tel']")
                total_input.clear()
                total_input.send_keys(str(total_amount))
                
                # STOP HERE if requested
                if stop_at_confirmation:
                    print("Stopping at Total Amount Input screen as requested.")
                    self._save_snapshot("stopped_at_confirmation")
                    return True, "確認画面で停止しました（シミュレーション成功）"
                
                # 7. 最終投票 (Final Vote)
                vote_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), '投票')]")
                print("Clicking Final Vote button...")
                vote_btn.click()
                
                return True, "投票完了（シミュレーション）"

            except Exception as e:
                print(f"Final Vote sequence failed: {e}")
                with open("debug_final_vote_fail.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                return False, f"最終投票処理エラー: {e}"

        except Exception as e:
            print(f"Vote Error: {e}")
            self._save_debug_screenshot(self.driver, "vote_error")
            return False, f"投票処理エラー: {e}"

    def close(self):
        if self.driver:
            self.driver.quit()
