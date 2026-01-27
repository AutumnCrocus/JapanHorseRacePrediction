
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
from typing import List, Dict, Any
import os
import datetime

class NetkeibaAutomator:
    def __init__(self):
        pass

    def _save_debug_screenshot(self, driver, name):
        """デバッグ用スクリーンショットを保存"""
        try:
            debug_dir = os.path.join(os.getcwd(), "scripts", "debug_screenshots")
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            path = os.path.join(debug_dir, f"{timestamp}_{name}.png")
            driver.save_screenshot(path)
            print(f"  [DEBUG] Screenshot saved: {path}")
        except Exception as e:
            print(f"  [DEBUG] Failed to save screenshot: {e}")

    def open_ipat_page(self, race_id):
        """IPAT連携ページを開いてドライバを返す"""
        print(f"=== Opening IPAT Page for Race ID: {race_id} ===")
        
        # オプション設定
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True) 
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        print("Initializing Chrome Driver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        try:
            # ウィンドウを最大化
            driver.maximize_window()
            print("Chrome Driver initialized successfully.")

            # 1. 出馬表ページにアクセス
            step_prefix = "step1"
            print(f"\nStep 1: Navigating to Shutuba (race card) page...")
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            print(f"URL: {url}")
            driver.get(url)
            time.sleep(2)
            self._save_debug_screenshot(driver, f"{step_prefix}_shutuba_page")
            
            # 2. IPAT連携ボタンをクリック
            step_prefix = "step2"
            print(f"\nStep 2: Clicking 'IPAT連携' button...")
            
            # 現在のウィンドウハンドルを保存
            original_window = driver.current_window_handle
            print(f"  Original window handle: {original_window}")
            
            # A. まずはURLを構築（これが最も確実）
            ipat_base_url = f"https://race.netkeiba.com/ipat/dispatch.html?race_id={race_id}"
            print(f"  [INFO] Constructing IPAT URL: {ipat_base_url}")

            # B. ボタンを探してみる（念のため）
            ipat_btn = None
            try:
                selectors = [
                    "a.IpatRenkeiBtn",
                    ".IpatRenkeiBtn",
                    "//a[contains(@class, 'IpatRenkeiBtn')]",
                    "//span[contains(text(), 'IPAT連携')]/.."
                ]
                for selector in selectors:
                    try:
                        by = By.XPATH if "//" in selector else By.CSS_SELECTOR
                        ipat_btn = driver.find_element(by, selector) # 待機なしで即時チェック
                        if ipat_btn: break
                    except: continue
                
                if ipat_btn:
                    href = ipat_btn.get_attribute("href")
                    if href:
                        print(f"  [INFO] Found actual button URL: {href}")
                        ipat_base_url = href # ボタンのURLがあればそちらを優先
            except: pass

            # C. 新しいタブで開く (window.openを使用することで window_handles が確実に増える)
            print(f"  [INFO] Opening IPAT URL in new tab: {ipat_base_url}")
            driver.execute_script(f"window.open('{ipat_base_url}', '_blank');")
            
            # D. 新しいウィンドウを検知して切り替え
            print("  Waiting for IPAT window to open (max 60s)...")
            ipat_window = None
            start_time = time.time()
            
            while time.time() - start_time < 60:
                current_handles = driver.window_handles
                for handle in current_handles:
                    if handle == original_window: continue
                    try:
                        driver.switch_to.window(handle)
                        # URLに ipat または dispatch が含まれているか確認
                        if "ipat" in driver.current_url or "dispatch" in driver.current_url:
                            ipat_window = handle
                            break
                    except: pass
                
                if ipat_window: break
                
                # まだ見つからなければ元に戻して少し待つ
                driver.switch_to.window(original_window)
                time.sleep(1)
            
            if not ipat_window:
                print("  [ERROR] Timed out waiting for IPAT window via window.open.")
                raise Exception("IPAT window not opened")

            # 切り替え確定
            driver.switch_to.window(ipat_window)
            print(f"  [OK] Switch to IPAT window: {driver.current_url}")

            # E. もしdispatch画面なら、自動遷移を待つか、リダイレクトが行われるはず
            # 必要であればここで待機ロジックを入れるが、通常は勝手にipat.htmlに行く
            
            time.sleep(2)
            self._bring_window_to_front(driver) 
            self._save_debug_screenshot(driver, f"{step_prefix}_ipat_page_initial")
            
            return driver
            
        except Exception as e:
            print(f"Error opening IPAT page: {e}")
            try:
                os.makedirs("scripts/debug_screenshots", exist_ok=True)
                with open("scripts/debug_screenshots/open_page_error.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                driver.save_screenshot("scripts/debug_screenshots/open_page_error.png")
            except:
                pass
            driver.quit()
            raise

    def launch_browser_for_ipat(self, race_id: str, bets: List[Dict[str, Any]]):
        """
        IPAT投票ページに直接アクセスし、買い目を入力してIPAT連携を行う。
        """
        print(f"=== Launching IPAT Direct Input Automation ===")
        print(f"Race ID: {race_id}")
        print(f"Total Bets Received: {len(bets)}")
        
        # ベット配列の内容をログ出力
        if not bets:
            print("WARNING: No bets provided. Browser will open but no automation will run.")
        else:
            print("Bets detail:")
            for idx, bet in enumerate(bets):
                print(f"  Bet {idx+1}: type={bet.get('type')}, horse_no={bet.get('horse_no')}, amount={bet.get('amount')}")
        
        try:
            # ページを開く（リファクタリングしたメソッドを使用）
            driver = self.open_ipat_page(race_id)
            
            # 3. IPAT投票ページで買い目を入力
            print(f"\nStep 3: Entering bets on IPAT page...")
            
            success_count = 0
            
            for idx, bet in enumerate(bets, 1):
                b_type = int(bet.get('type', 1))
                method = bet.get('method', '通常')  # デフォルトは通常
                h_val = bet.get('horse_no')
                amount = int(bet.get('amount', 100))
                
                # 馬番のパース（辞書型の場合はそのまま使う）
                if isinstance(h_val, dict):
                    horses = h_val
                else:
                    horses = self._parse_horse_no(h_val)
                    
                if not horses:
                    print(f"  Bet {idx}: Invalid horse_no: {h_val}, skipping")
                    continue
                
                # 券種名を取得
                type_name = self._get_bet_type_name(b_type)
                print(f"\n  --- Bet {idx}: {type_name} ({method}), Horses: {horses}, Amount: {amount}円 ---")
                
                try:
                    # 1. 券種タブを選択
                    self._select_ticket_type_tab(driver, b_type)
                    time.sleep(0.5)
                    self._save_debug_screenshot(driver, f"03_bet{idx}_tab_selected")
                    
                    # 2. 方式を選択 (通常, フォーメーション, etc)
                    self._select_betting_method(driver, method)
                    time.sleep(0.5)
                    self._save_debug_screenshot(driver, f"03_bet{idx}_method_selected")
                    
                    # 3. 馬番を入力
                    self._enter_horse_numbers(driver, horses)
                    time.sleep(0.5)
                    self._save_debug_screenshot(driver, f"04_bet{idx}_horse_selected")
                    
                    # 金額を入力（100円単位）
                    self._enter_amount(driver, amount)
                    time.sleep(0.5)
                    self._save_debug_screenshot(driver, f"05_bet{idx}_amount_entered")
                    
                    # 追加ボタンをクリック
                    self._click_add_button(driver)
                    time.sleep(1.0)
                    self._save_debug_screenshot(driver, f"06_bet{idx}_added")
                    
                    print(f"    [SUCCESS] Added bet: {type_name} {horses} - {amount}円")
                    success_count += 1
                    
                except Exception as e:
                    # アラートが出た場合は、入力不備の可能性があるが、ユーザーに手動で任せるためにエラーにはしない
                    error_msg = str(e)
                    if "UnexpectedAlertPresentException" in error_msg or "unexpected alert open" in error_msg:
                        print(f"    [INFO] Alert detected! Likely due to validation error.")
                        try:
                            # アラートを閉じる
                            driver.switch_to.alert.accept()
                            print("    [INFO] Alert accepted.")
                        except:
                            pass
                        print("    [WARN] Please complete this bet manually.")
                        # スクリーンショットを撮ってみる（アラート後）
                        self._save_debug_screenshot(driver, f"07_bet{idx}_alert_handled")
                    else:
                        print(f"    [ERROR] Failed to add bet: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\nTotal bets successfully added: {success_count}/{len(bets)}")
            
            # 4. IPAT投票ボタンの表示を確認（オプション）
            if success_count > 0:
                print("\nStep 4: Checking for IPAT vote button...")
                try:
                    ipat_btn = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, "ipat_dialog"))
                    )
                    print("  [OK] IPAT投票ボタンが表示されています")
                    print("  ユーザーは手動でボタンをクリックして投票を完了してください")
                except Exception as e:
                    print(f"  [INFO] IPAT投票ボタンが見つかりません（正常な場合もあります）")
            else:
                print("\nNo bets were added. Please check the logs above.")

        except Exception as e:
            print(f"Automation Error: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                # アラートがあれば閉じる
                try:
                    alert = driver.switch_to.alert
                    print(f"Alert found: {alert.text}")
                    alert.accept()
                except:
                    pass
                
                # DOMを保存して解析できるようにする
                os.makedirs("scripts/debug_screenshots", exist_ok=True)
                with open("scripts/debug_screenshots/crash_dom.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print("  [DEBUG] Saved crash_dom.html")
                self._save_debug_screenshot(driver, "crash_screenshot")
            except Exception as save_err:
                print(f"  [DEBUG] Failed to save crash info: {save_err}")
                
        finally:
            # 処理完了後、ブラウザウィンドウを最前面に表示
            try:
                driver.switch_to.window(driver.current_window_handle)
                
                # Windowsでブラウザを最前面に表示
                try:
                    import win32gui
                    import win32con
                    # Chromeのウィンドウハンドルを取得
                    hwnd = driver.current_window_handle
                    # 最前面に表示
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                         win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                         win32con.SWP_SHOWWINDOW | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                except:
                    # win32guiが使えない場合はJavaScriptで試行
                    driver.execute_script("window.focus();")
                
                print("\n=== Automation Complete ===")
                print("ブラウザが最前面に表示されました。投票手続きを完了してください。")
            except:
                pass

    def _parse_horse_no(self, val):
        """馬番入力値([1] or "1-2" etc)を整数のリストに変換"""
        try:
            if isinstance(val, int):
                return [val]
            if isinstance(val, list):
                return [int(x) for x in val]
            if isinstance(val, str):
                if '-' in val:
                    return [int(x) for x in val.split('-')]
                if val.isdigit():
                    return [int(val)]
        except:
            pass
        return []

    def _get_bet_selector(self, type_code, horses: List[int]):
        """
        券種と馬番リストから、投票用チェックボックスのセレクタを返す。
        戻り値: (By.ID or By.CSS_SELECTOR, selector_string)
        """
        if not horses:
            return None
            
        # 単勝(1), 複勝(2) -> 通常投票のチェックボックス
        if type_code in [1, 2]:
            # HTMLダンプから判明: id="uc-0-{horse_no}" でチェックボックスを特定
            # value属性は全て "1" なので使用不可
            return (By.ID, f"uc-0-{horses[0]}")
            
        # 枠連(3), 馬連(4), ワイド(5), 馬単(6), 3連複(7), 3連単(8)
        if type_code in [3, 4, 5, 6, 7, 8]:
            target_horses = horses
            # 馬番の並び順ルール: 馬連・ワイド・3連複 -> 昇順
            if type_code in [3, 4, 5, 7]:
                target_horses = sorted(horses)
                
            # 通常投票の場合、リスト内の馬番を順にチェックするだけなので
            # 単複と同じセレクタで1頭目のチェックボックスを返す（呼び出し元でループ処理される前提なら）
            # ただし、呼び出し元が _enter_horse_numbers_simple の場合、horses リストの各要素に対して呼ばれる
            if len(horses) == 1:
                 return (By.ID, f"uc-0-{horses[0]}")
            
            # 念のため、これまで通りのロジックも残すが、基本は上記のセレクタで良いはず
            # 軸・相手指定などで複雑なIDになる場合は別途対応が必要（Header baseで処理されるためここは通らないはず）

            
            # 1_2_3 のような形式を作成
            h_suffix = "_".join([str(h) for h in target_horses])
            
            # CSSセレクタで末尾一致検索: input[id$='_b4_c0_1_2']
            return (By.CSS_SELECTOR, f"input[id$='_b{type_code}_c0_{h_suffix}']")

        return None
    
    def _get_bet_type_name(self, type_code):
        """券種コードから券種名を取得"""
        type_names = {
            1: '単勝',
            2: '複勝',
            3: '枠連',
            4: '馬連',
            5: 'ワイド',
            6: '馬単',
            7: '三連複',
            8: '三連単'
        }
        return type_names.get(type_code, f'Unknown({type_code})')
    
    def _select_ticket_type_tab(self, driver, type_code):
        """券種タブを選択"""
        tab_map = {
            1: '単勝',
            2: '複勝',
            3: '枠連',
            4: '馬連',
            5: 'ワイド',
            6: '馬単',
            7: '3連複',
            8: '3連単'
        }
        
        tab_name = tab_map.get(type_code)
        if not tab_name:
            raise ValueError(f"Unknown ticket type: {type_code}")
        
        print(f"    Selecting ticket type tab: {tab_name}")
        
        # タブをクリック（XPathでテキストマッチ）
        try:
            tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//a[contains(text(), '{tab_name}')]"))
            )
            driver.execute_script("arguments[0].click();", tab)
            print(f"    [OK] Tab selected: {tab_name}")
            
            # JavaScriptによるDOM更新を待機（延長）
            time.sleep(2.0)
            
        except Exception as e:
            print(f"    [ERROR] Failed to select tab {tab_name}: {e}")
            raise
    
    
    def _select_betting_method(self, driver, method_name):
        """
        投票方式（通常、フォーメーション、ボックス、ながし）を選択する
        """
        print(f"    Selecting betting method: {method_name}")
        try:
            # 1. まず <dl class="houshiki"> 内を探す (新しいデザイン?)
            # <dl class="houshiki"><dt>方式選択</dt><dd class="Active"><a href="...">通常</a></dd>...</dl>
            xpath = f"//dl[contains(@class, 'houshiki')]//a[contains(text(), '{method_name}')]"
            
            method_tab = None
            try:
                method_tab = driver.find_element(By.XPATH, xpath)
            except:
                # 古いデザインや別パターンの場合 (MethodNav)
                xpath = f"//ul[contains(@class, 'Method')]//a[contains(text(), '{method_name}')]"
                try:
                    method_tab = driver.find_element(By.XPATH, xpath)
                except:
                    pass

            if not method_tab:
                print(f"    [WARN] Method tab '{method_name}' not found.")
                return 

            # 既にアクティブか確認
            # 親要素 (dd or li) が Active クラスを持っているか
            parent = method_tab.find_element(By.XPATH, "./..")
            parent_class = parent.get_attribute("class")
            
            if parent_class and "Active" in parent_class:
                print(f"    [INFO] Method {method_name} is already active.")
            else:
                # クリック
                driver.execute_script("arguments[0].click();", method_tab)
                print(f"    [OK] Selected method: {method_name}")
                time.sleep(1.0) # UI切り替え待機
                
        except Exception as e:
            print(f"    [WARN] Failed to select betting method '{method_name}': {e}")
            import traceback
            traceback.print_exc()
            # エラーにはせず続行（デフォルトが通常なので、通常の場合はエラーでもOKな場合が多い）
            if method_name != "通常":
                raise e

    def _bring_window_to_front(self, driver):
        """ブラウザウィンドウを最前面に表示する"""
        try:
            # 1. Selenium標準の切り替え
            current_handle = driver.current_window_handle
            driver.switch_to.window(current_handle)
            
            # 2. JavaScriptによるフォーカス
            driver.execute_script("window.focus();")
            
            # 3. 最小化 -> 最大化による強制前面表示（ユーザー要望により復活）
            # IPAT画面遷移直後のみ有効に動作することを期待
            try:
                driver.minimize_window()
                time.sleep(0.1)
                driver.maximize_window()
                time.sleep(0.1)
                print("    [INFO] Cycled window minimize/maximize to bring to front.")
            except:
                pass
            
        except Exception as e:
            print(f"    [WARN] Failed to focus window: {e}")

    def _enter_horse_numbers(self, driver, horses):
        """
        馬番を選択する。
        horsesがリスト([1,2])なら通常/ボックス投票。
        horsesが辞書({'1着': [1], '2着': [2,3]})ならヘッダーベースの入力を行う。
        """
        if isinstance(horses, dict):
            self._enter_horse_numbers_by_header(driver, horses)
        else:
            self._enter_horse_numbers_simple(driver, horses)

    def _enter_horse_numbers_simple(self, driver, horses):
        """従来の馬番選択（通常・ボックス用）"""
        print(f"    Selecting horse numbers (Simple): {horses}")
        
        for horse_no in horses:
            try:
                # 行を特定
                xpath = f"//tr[td[position()=2 and normalize-space(text())='{horse_no}']]"
                horse_row = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                
                # 行内の最初のチェックボックスまたはIDがuc-0-Xみたいなものを探す
                # 通常投票なら行内にチェックボックスは1つか、単勝/複勝で分かれている
                checkboxes = horse_row.find_elements(By.TAG_NAME, "input")
                target_cb = None
                
                for cb in checkboxes:
                    cb_type = cb.get_attribute("type")
                    cb_class = cb.get_attribute("class") or ""
                    
                    # type=checkboxで、HorseCheck_Selectクラスを持つものを優先
                    if cb_type == "checkbox" and "HorseCheck_Select" in cb_class:
                        target_cb = cb
                        break
                    # フォールバック: type=checkboxなら対象
                    elif cb_type == "checkbox":
                        target_cb = cb
                
                if target_cb:
                    # 既にチェックされているか確認
                    if target_cb.is_selected():
                         print(f"    [INFO] Horse #{horse_no} is already checked.")
                    else:
                        print(f"    [DEBUG] Before click: {target_cb.get_attribute('outerHTML')}")
                        
                        # 戦略: label -> parent td -> js click の順に試す
                        # netkeibaのチェックボックスは label でラップされているか、for属性付きのlabelがある可能性が高い
                        click_success = False
                        
                        # 1. Labelを探してクリック
                        try:
                            # inputのIDを取得
                            cb_id = target_cb.get_attribute("id")
                            if cb_id:
                                label = driver.find_elements(By.CSS_SELECTOR, f"label[for='{cb_id}']")
                                if label:
                                    print(f"    [INFO] Clicking LABEL for horse #{horse_no}...")
                                    label[0].click()
                                    click_success = True
                        except Exception as e:
                            print(f"    [WARN] Click label failed: {e}")

                        # 2. 親要素(TD)をクリック (Labelで失敗した場合)
                        if not click_success:
                            try:
                                parent_td = target_cb.find_element(By.XPATH, "./..")
                                print(f"    [INFO] Clicking parent TD for horse #{horse_no}...")
                                parent_td.click()
                                click_success = True
                            except Exception as e:
                                print(f"    [WARN] Click parent TD failed: {e}")

                        # 3. JSでinputをクリック (最終手段)
                        if not click_success:
                             driver.execute_script("arguments[0].click();", target_cb)

                        time.sleep(0.3)
                        
                        # 最終確認 (ループで待機)
                        start_wait = time.time()
                        checked = False
                        while time.time() - start_wait < 3.0: # 待機時間を少し延長
                            if target_cb.is_selected():
                                checked = True
                                break
                            try:
                                if target_cb.get_attribute("checked") == "true":
                                    checked = True
                                    break
                            except: pass
                            time.sleep(0.1)
                            
                        # Standard check
                        if not checked and target_cb.is_selected():
                            checked = True

                        if checked:
                            print(f"    [OK] Checked horse #{horse_no}")
                            print(f"    [DEBUG] After success: {target_cb.get_attribute('outerHTML')}")
                        else:
                            print(f"    [ERROR] Failed to check horse #{horse_no} after retries!")
                            print(f"    [DEBUG] After failure: {target_cb.get_attribute('outerHTML')}")
                            self._save_debug_screenshot(driver, f"error_check_failed_{horse_no}")
                            
                            # 強制JS設定とChangeイベント発火
                            print("    [WARN] Forcing state and firing change event...")
                            driver.execute_script("""
                                arguments[0].checked = true;
                                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                                arguments[0].dispatchEvent(new Event('click', { bubbles: true }));
                            """, target_cb)
                            time.sleep(0.5)
                            
                            if not target_cb.is_selected():
                                raise Exception(f"Failed to check horse #{horse_no}")
                else:
                    print(f"    [WARN] No checkbox found for horse #{horse_no}")
                    try:
                        debug_html_path = os.path.join(os.getcwd(), "scripts", "debug_screenshots", f"debug_source_step3_miss_{horse_no}.html")
                        with open(debug_html_path, "w", encoding="utf-8") as f:
                            f.write(driver.page_source)
                        print(f"    [DEBUG] Saved page source to {debug_html_path}")
                    except Exception as e:
                        print(f"    [WARN] Failed to save debug HTML: {e}")

                time.sleep(0.2)
                
            except Exception as e:
                print(f"    [ERROR] Failed to select horse #{horse_no}: {e}")
                continue

    def _enter_horse_numbers_by_header(self, driver, horse_map):
        """
        ヘッダーテキストに基づいて列を特定し、馬番を選択する
        horse_map: {'1頭目': [1], '2頭目': [2,3]} などの辞書
        """
        print(f"    Selecting horse numbers (Header-based): {horse_map}")
        
        # 1. ヘッダー情報の解析
        # thのテキストと列インデックスのマッピングを作成
        header_map = {} # {'1頭目': 3, '2頭目': 4, ...} (0-indexed relative to tr children)
        
        try:
            # ページ内のテーブルヘッダーを探す
            # 投票フォームのテーブルは通常 class="VoteTable" などの特徴があるが
            # ここでは汎用的に th を全部スキャンしてターゲットを探す
            
            headers = driver.find_elements(By.XPATH, "//table//th")
            for i, h in enumerate(headers):
                text = h.text.strip()
                if text:
                    # ヘッダーテキストをキーにする。改行などは除去
                    text = text.replace('\n', '')
                    # 既に同じテキストがある場合の処理（あまりないはずだが）
                    # ここでは単純に保存。実際の列インデックスはTDの位置と合わせる必要がある
                    pass

            # より正確なアプローチ: 
            # 各行のtrを取得し、その親のtableのtheadを見る
            
            # 馬番1の行を基準にしてテーブル構造を把握する
            xpath_sample = f"//tr[td[position()=2 and normalize-space(text())='1']]"
            sample_row = driver.find_element(By.XPATH, xpath_sample)
            parent_table = sample_row.find_element(By.XPATH, "./ancestor::table")
            
            # thead内のthを取得 (複数行ヘッダーの可能性もあるが、通常は最下行のthが列に対応)
            # trが複数ある場合、最後のtrを使う
            thead_rows = parent_table.find_elements(By.CSS_SELECTOR, "thead tr")
            if not thead_rows:
                # theadがない場合は最初のtr?
                header_row = parent_table.find_element(By.CSS_SELECTOR, "tr")
            else:
                header_row = thead_rows[-1]
            
            cols = header_row.find_elements(By.TAG_NAME, "th")
            # thがない場合はtdかも
            if not cols:
                cols = header_row.find_elements(By.TAG_NAME, "td")
                
            print(f"    Found {len(cols)} columns in header.")
            
            # マッピング作成
            # 列インデックスは 0-based
            valid_col_indices = {}
            for idx, col in enumerate(cols):
                txt = col.text.strip().replace('\n', '')
                print(f"      Col {idx}: {txt}")
                valid_col_indices[txt] = idx
                # 部分一致用にもキーを追加（'1頭目' -> '1頭' など）
                if '1頭' in txt: valid_col_indices['1頭目'] = idx
                if '2頭' in txt: valid_col_indices['2頭目'] = idx
                if '3頭' in txt: valid_col_indices['3頭目'] = idx
                if '軸' in txt: valid_col_indices['軸'] = idx
                if '相手' in txt: valid_col_indices['相手'] = idx
                if '1着' in txt: valid_col_indices['1着'] = idx
                if '2着' in txt: valid_col_indices['2着'] = idx
                if '3着' in txt: valid_col_indices['3着'] = idx
        
        except Exception as e:
            print(f"    [ERROR] Failed to parse table header: {e}")
            raise

        # 2. 各指定ごとの処理
        for key, horses in horse_map.items():
            target_col_idx = valid_col_indices.get(key)
            if target_col_idx is None:
                print(f"    [WARN] Header '{key}' not found in table. Available: {list(valid_col_indices.keys())}")
                continue
                
            print(f"    Processing '{key}' (Col {target_col_idx}) -> Horses: {horses}")
            
            for horse_no in horses:
                try:
                    # 行を再特定
                    xpath = f"//tr[td[position()=2 and normalize-space(text())='{horse_no}']]"
                    row = driver.find_element(By.XPATH, xpath)
                    
                    # その行のセルを取得
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # セル位置の調整が必要かもしれない
                    # 馬番セルがindex 1 (2番目) なので、ヘッダーの列数とセルの列数が一致しているか確認が必要
                    # Netkeibaの場合、ヘッダーとボディの列構成は一致しているはず
                    
                    if target_col_idx < len(cells):
                        target_cell = cells[target_col_idx]
                        
                        # チェックボックスを探してクリック
                        cb = target_cell.find_element(By.TAG_NAME, "input")
                        
                        if cb.is_selected():
                            print(f"      [INFO] Horse #{horse_no} in col '{key}' already checked.")
                        else:
                            print(f"      [DEBUG] Before click (Header): {cb.get_attribute('outerHTML')}")
                            # 戦略: Label -> Parent -> JS
                            click_success = False
                            
                            # 1. Label
                            try:
                                cb_id = cb.get_attribute("id")
                                if cb_id:
                                    label = driver.find_elements(By.CSS_SELECTOR, f"label[for='{cb_id}']")
                                    if label:
                                        print(f"      [INFO] Clicking LABEL for horse #{horse_no} in '{key}'...")
                                        label[0].click()
                                        click_success = True
                            except: pass

                            # 2. Parent TD
                            if not click_success:
                                try:
                                    print(f"      [INFO] Clicking parent TD for horse #{horse_no} in '{key}'...")
                                    target_cell.click() 
                                    click_success = True
                                except Exception as e:
                                    print(f"      [WARN] Click parent TD failed: {e}")
                            
                            # 3. JS
                            if not click_success:
                                driver.execute_script("arguments[0].click();", cb)
                            
                            time.sleep(0.3)
                            
                            # 最終確認 (ループで待機)
                            start_wait = time.time()
                            checked = False
                            while time.time() - start_wait < 3.0:
                                if cb.is_selected():
                                    checked = True
                                    break
                                time.sleep(0.1)

                            if checked:
                                print(f"      [OK] Checked horse #{horse_no} in col '{key}'")
                                try: print(f"      [DEBUG] After success: {cb.get_attribute('outerHTML')}")
                                except: pass
                            else:
                                print(f"      [ERROR] FAILED to check horse #{horse_no} in col '{key}'")
                                try: print(f"      [DEBUG] After failure: {cb.get_attribute('outerHTML')}")
                                except: pass
                                self._save_debug_screenshot(driver, f"error_check_failed_{horse_no}_{key}")
                                # 強制設定 + イベント発火
                                print("      [WARN] Forcing state and firing events...")
                                driver.execute_script("""
                                    arguments[0].checked = true;
                                    arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                                    arguments[0].dispatchEvent(new Event('click', { bubbles: true }));
                                """, cb)
                                time.sleep(0.2)
                                if not cb.is_selected():
                                    raise Exception(f"Failed to check horse #{horse_no} in col '{key}'")
                    else:
                        print(f"      [ERROR] Column index {target_col_idx} out of range (cells={len(cells)})")
                        
                except Exception as e:
                    print(f"      [ERROR] Failed to select horse #{horse_no} for '{key}': {e}")



    def _enter_amount(self, driver, amount):
        """金額を入力（100円単位）"""
        
        amount_in_hundreds = amount // 100
        print(f"    Entering amount: {amount}円 ({amount_in_hundreds} x 100円)")
        
        # デバッグ用ページソース保存は省略（安定したら削除可）
        
        amount_input = None
        
        # 1. 既知のクラス名と属性で探す
        selectors = [
            "input[name='money']", # IPAT連携で確認された正しいセレクタ
            "input.IpatVote_Amount",
            "input[name*='amount']",
            "input[id*='amount']", 
            "input.v-PatternAmount",
            "input[type='number']"
        ]
        
        # ... (中略: JS探索などはそのまま維持したいが、input[name='money']で確定ならシンプルにしてもいい)
        # ここでは既存のロジックを維持しつつ、selectorsの優先度変更のみ反映済なので、コード簡略化版を置く
        
        for selector in selectors:
            try:
                inputs = driver.find_elements(By.CSS_SELECTOR, selector)
                visible_inputs = [i for i in inputs if i.is_displayed()]
                if visible_inputs:
                    amount_input = visible_inputs[0]
                    print(f"    [Found] Input using selector: {selector}")
                    break
            except:
                continue
        
        if amount_input:
            try:
                # フォーカス＆クリア＆入力
                driver.execute_script("arguments[0].focus();", amount_input)
                amount_input.clear()
                amount_input.send_keys(str(amount_in_hundreds))
                
                # イベント発火
                driver.execute_script("""
                    arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
                    arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                    arguments[0].dispatchEvent(new Event('blur', { bubbles: true }));
                """, amount_input)
                
                print(f"    [OK] Entered amount: {amount_in_hundreds}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    [ERROR] Failed to enter amount: {e}")
                raise
        else:
            print("    [ERROR] Amount input field not found!")
            self._save_debug_screenshot(driver, "error_amount_input_not_found")
            raise Exception("Amount input field not found")

    def _click_add_button(self, driver):
        """追加ボタンをクリック"""
        print(f"    Clicking add button...")
        try:
            # 「追加」というテキストを持つボタンを探す (XPath)
            # input[name='money'] の親要素の周辺にあるはず
            
            add_btns = driver.find_elements(By.XPATH, "//button[contains(text(), '追加')]")
            visible_btns = [b for b in add_btns if b.is_displayed()]
            
            if visible_btns:
                # 複数ある場合は最初のもの（通常投票エリアのもの）
                btn = visible_btns[0]
                driver.execute_script("arguments[0].click();", btn)
                print("    [OK] '追加' button clicked via XPath")
            else:
                # クラス名でのフォールバック
                print("    [INFO] '追加' text button not found, trying CSS selector .AddBtn")
                btn = driver.find_element(By.CSS_SELECTOR, "button.AddBtn, .AddBtn")
                driver.execute_script("arguments[0].click();", btn)
                print("    [OK] Add button clicked via CSS")
                
            time.sleep(1.0)
            
            # アラートが出た場合は入力検証エラー
            try:
                alert = driver.switch_to.alert
                alert_text = alert.text
                print(f"    [ERROR] Alert detected: {alert_text}")
                alert.accept()  # アラートを閉じる
                raise Exception(f"Validation alert: {alert_text}")
            except Exception as alert_ex:
                # アラートがない場合は正常（NoAlertPresentExceptionなど）
                if "alert" not in str(alert_ex).lower() and "unexpected" not in str(alert_ex).lower():
                    # 想定外のエラーなら再投げ
                    raise
                # アラートがあった場合は既にraiseしている
                if "Validation alert" in str(alert_ex):
                    raise
            
        except Exception as e:
            print(f"    [ERROR] Failed to click add button: {e}")
            self._save_debug_screenshot(driver, "error_add_button")
            raise
