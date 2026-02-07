"""
IPAT直接連携モジュール（Selenium版 - PC Site）
JRA IPAT (PC版) にアクセスして投票画面を自動操作するモジュール
URL: https://www.ipat.jra.go.jp/
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys

import time
from typing import List, Dict, Any, Optional
import os
import datetime
import traceback
import sys
import platform

class IpatDirectAutomator:
    """IPAT直接連携クラス（Selenium版 - PC Site）"""
    
    # 定数
    JRA_IPAT_URL = "https://www.ipat.jra.go.jp/" # PC版URL
    WAIT_SEC = 0.5  # 基本待機時間
    WAIT_SEC_LONG = 1.0  # ページ遷移時の待機時間
    
    # 曜日リスト
    DOW_LST = ["月", "火", "水", "木", "金", "土", "日"]
    # レース会場リスト (netkeibaの表記とIPATの表記のマッピングが必要な場合に備える)
    # PC版でも基本は似ているが、画面上のテキストとのマッチングに使用
    PLACE_LST = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    
    def __init__(self, debug_mode: bool = False):
        """初期化"""
        self.driver = None
        self.wait_timeout = 10
        self.debug_mode = debug_mode
        
    def _save_debug_screenshot(self, driver, name: str):
        """デバッグ用スクリーンショットを保存（debug_mode=Trueの場合のみ）"""
        if not self.debug_mode:
            return
            
        try:
            screenshot_dir = os.path.join(os.getcwd(), 'debug_screenshots')
            os.makedirs(screenshot_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(screenshot_dir, f'{name}_{timestamp}.png')
            driver.save_screenshot(filepath)
            print(f"Screenshot saved: {filepath}")
        except Exception as e:
            print(f"Failed to save screenshot: {e}")

    def _setup_driver(self):
        """WebDriverをセットアップする (PC版設定)"""
        options = Options()
        
        # PC版なのでモバイルエミュレーションは削除
        # mobile_emulation = { "deviceName": "iPhone X" }
        # options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # PC用ウィンドウサイズ
        options.add_argument("--window-size=1280,800")
        options.add_argument("--lang=ja")
        
        # 自動化検出回避 (基本設定)
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Headlessモードは無効化 (ユーザー確認が必要なため)
        
        # WebDriverの初期化
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(self.wait_timeout)

    def login(self, inetid: str, subscriber_no: str, pin: str, pars_no: str) -> tuple[bool, str]:
        """
        IPATログイン画面で認証を実行（PC版）
        
        Args:
            inetid: INET-ID
            subscriber_no: 加入者番号
            pin: 暗証番号
            pars_no: P-ARS番号
        """
        try:
            # ブラウザが起動していなければ起動
            if not self.driver:
                self._setup_driver()
            
            # 既にログイン済みかチェック
            try:
                current_url = self.driver.current_url
                # PC版メニュー画面URLパターン (pw_020_i.cgi など)
                if "pw_" in current_url and "020" in current_url: 
                    print("Already logged in (URL match). Skipping login sequence.")
                    return True, "ログイン済み"
                
                # タイトルや要素でのチェック
                if "I-PAT" in self.driver.title or "投票メニュー" in self.driver.page_source:
                     # ログアウトボタン等があればログイン済みと判定
                     logout_btn = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'ログアウト')]")
                     if logout_btn:
                        print("Already logged in (Logout button found). Skipping login sequence.")
                        return True, "ログイン済み"
            except:
                pass
            
            print(f"IPAT(PC)へアクセス中: {self.JRA_IPAT_URL}")
            self.driver.get(self.JRA_IPAT_URL)
            time.sleep(self.WAIT_SEC)
            
            # トップページに「ログイン」ボタンがある場合（ランディングページ）の処理
            # またはフレーム構造の対応
            
            # フレーム対応: メインコンテンツが含まれるフレームを探す
            frames = self.driver.find_elements(By.TAG_NAME, "frame") # iframeではなくframeタグの場合が多い
            if not frames:
                frames = self.driver.find_elements(By.TAG_NAME, "iframe")
                
            if len(frames) > 0:
                print(f"Frames found: {len(frames)}. Searching for login form inside frames...")
                found_frame = False
                for i, frame in enumerate(frames):
                    self.driver.switch_to.default_content()
                    try:
                        self.driver.switch_to.frame(frame)
                        if len(self.driver.find_elements(By.NAME, "i")) > 0 or len(self.driver.find_elements(By.NAME, "inetid")) > 0:
                            print(f"Login form found in frame index {i}")
                            found_frame = True
                            # switch_to.frameしたまま抜ける
                            break
                        # ログインボタンがあるか？
                        if len(self.driver.find_elements(By.XPATH, "//a[contains(@class, 'login')]")) > 0:
                             print(f"Login button found in frame index {i}")
                             found_frame = True
                             break
                    except:
                        pass
                
                if not found_frame:
                    self.driver.switch_to.default_content()

            try:
                # ログインボタンを探して押す
                # PC版は 'inetid' 入力欄がいきなりある場合と、ボタンの場合がある
                # すでにフレームに入っている場合はその中で探す
                if len(self.driver.find_elements(By.NAME, "inetid")) == 0 and len(self.driver.find_elements(By.NAME, "i")) == 0:
                     print("Login form not found directly. Searching for entry button...")
                     entry_btn_candidates = [
                         "//a[contains(@class, 'login')]",
                         "//img[contains(@alt, 'ログイン') or contains(@alt, 'Log in')]/..",
                         "//a[contains(text(), 'ログイン')]",
                         "//input[@value='ログイン']"
                     ]
                     for xpath in entry_btn_candidates:
                         try:
                             btn = self.driver.find_element(By.XPATH, xpath)
                             btn.click()
                             time.sleep(self.WAIT_SEC_LONG)
                             break
                         except:
                             continue
            except:
                pass
            
            # ユーザー提供情報による2段階ログインフロー
            # 1. INET-ID 入力 -> ログインボタン
            # 2. 加入者情報入力 (加入者番号, 暗証番号, P-ARS番号) -> ネット投票メニューへ (ログインボタン)
            
            try:
                 # --- STEP 1: INET-ID 入力 ---
                 print("Login Step 1: INET-ID")
                 
                 # inetid入力フィールドを探す
                 # フレーム切り替え等はここまでの処理で実施済み
                 try:
                     inet_elem = self.driver.find_element(By.NAME, "inetid")
                 except:
                     # 見つからない場合はフレーム探索などが失敗している可能性
                     # 再度フレームを探してみる
                     frames = self.driver.find_elements(By.TAG_NAME, "frame")
                     if not frames: frames = self.driver.find_elements(By.TAG_NAME, "iframe")
                     for f in frames:
                         try:
                             self.driver.switch_to.default_content()
                             self.driver.switch_to.frame(f)
                             inet_elem = self.driver.find_element(By.NAME, "inetid")
                             print("INET-ID input found in frame.")
                             break
                         except:
                             pass
                 
                 # 入力
                 try:
                     inet_elem = self.driver.find_element(By.NAME, "inetid")
                     inet_elem.clear()
                     inet_elem.send_keys(inetid)
                     
                     # ログインボタン (INET-ID画面)
                     # ボタンを押して次へ
                     # 画像ボタンやsubmitボタンの可能性がある
                     # ユーザー画像では「ログイン」というボタンが見える
                     step1_btn_candidates = [
                         "//a[contains(text(), 'ログイン')]",
                         "//input[@value='ログイン']",
                         "//img[contains(@alt, 'ログイン')]/..",
                         "//a[contains(@onclick, 'Login')]"
                     ]
                     
                     clicked_step1 = False
                     for xpath in step1_btn_candidates:
                         try:
                             btn = self.driver.find_element(By.XPATH, xpath)
                             btn.click()
                             clicked_step1 = True
                             break
                         except:
                             continue
                     
                     if not clicked_step1:
                         # Enterキーでsubmit試行
                         inet_elem.send_keys(Keys.RETURN)
                     
                     time.sleep(self.WAIT_SEC_LONG)
                     
                 except Exception as e:
                     print(f"Step 1 (INET-ID) failed: {e}")
                     # 既にStep 2の画面にいる可能性もあるので続行してみる

                 # --- STEP 2: 加入者情報入力 ---
                 print("Login Step 2: Subscriber Info")
                 
                 # 画面が変わったことを想定して要素を探す
                 # 加入者番号: i, 暗証番号: p, P-ARS: r
                 
                 # 加入者番号 (i)
                 try:
                     i_elem = self.driver.find_element(By.NAME, "i")
                     i_elem.clear()
                     i_elem.send_keys(subscriber_no)
                 except:
                     try:
                         # フレームが変わった可能性、またはまだ読込中
                         time.sleep(1.0)
                         frames = self.driver.find_elements(By.TAG_NAME, "frame")
                         if not frames: frames = self.driver.find_elements(By.TAG_NAME, "iframe")
                         for f in frames:
                             self.driver.switch_to.default_content()
                             self.driver.switch_to.frame(f)
                             if len(self.driver.find_elements(By.NAME, "i")) > 0:
                                 break
                         i_elem = self.driver.find_element(By.NAME, "i")
                         i_elem.clear()
                         i_elem.send_keys(subscriber_no)
                     except:
                         print("Subscriber No input not found.")
                 
                 # 暗証番号 (p)
                 try:
                     p_elem = self.driver.find_element(By.NAME, "p")
                     p_elem.clear()
                     p_elem.send_keys(pin)
                 except:
                     print("PIN input not found.")
                     
                 # P-ARS番号 (r)
                 try:
                     r_elem = self.driver.find_element(By.NAME, "r")
                     r_elem.clear()
                     r_elem.send_keys(pars_no)
                 except:
                     print("P-ARS input not found.")
                 
                # ログインボタン (ネット投票メニューへ)
                 # ユーザー画像では「ネット投票メニューへ」
                 # HTML解析結果: <a href="#" onclick="JavaScript:ToModernMenu();" title="ネット投票メニューへ"></a>
                 step2_btn_candidates = [
                     "//a[@title='ネット投票メニューへ']",
                     "//a[contains(@onclick, 'ToModernMenu')]",
                     "//a[contains(text(), 'ネット投票メニューへ')]", # Fallback
                     "//a[contains(text(), 'ログイン')]"         # Fallback
                 ]
                  
                 clicked_step2 = False
                 for xpath in step2_btn_candidates:
                     try:
                         print(f"Trying Step2 Btn: {xpath}")
                         btn = self.driver.find_element(By.XPATH, xpath)
                         # 可視状態チェック
                         if btn.is_displayed():
                             btn.click()
                             print("  -> Clicked!")
                             clicked_step2 = True
                             break
                         else:
                             print("  -> Found but invisible.")
                             # JSでクリックトライ
                             self.driver.execute_script("arguments[0].click();", btn)
                             print("  -> JS Clicked!")
                             clicked_step2 = True
                             break
                     except Exception as ex:
                         print(f"  -> Failed: {ex}")
                         continue
                  
                 if not clicked_step2:
                     print("Step2: All buttons failed. Executing JS directly.")
                     # JS実行
                     self.driver.execute_script("try { ToModernMenu(); } catch(e) { try { DoLogin(); } catch(ex) {} }")
                  
                 # 遷移待ち (URL変化 or 特定要素出現)
                 time.sleep(self.WAIT_SEC)
                 try:
                     WebDriverWait(self.driver, 10).until(lambda d: "pw_890" in d.current_url or "netkeiba" in d.current_url)
                     print("Step2: URL transition confirmed.")
                 except:
                     print("Step2: URL transition timed out.")
            
            except Exception as e:
                print(f"Login sequence failed: {e}")
                if not self.debug_mode:
                    pass
            
            # ログイン成功確認
            time.sleep(1.0)
            current_url = self.driver.current_url
            # 「加入者番号」入力欄が消えていることを確認 (重要)
            is_still_login_page = "加入者番号" in self.driver.page_source and "input" in self.driver.page_source
            
            # 成功判定: URLがメニュー画面 (pw_890) になっていること
            if "pw_890" in current_url:
                print("Login success (URL Verified).")
                return True, "ログイン成功"
            elif not is_still_login_page and ("ネット投票" in self.driver.page_source or "投票メニュー" in self.driver.page_source):
                # URLが変わらなくても画面が変わっていればOK
                print("Login success (Content Verified).")
                return True, "ログイン成功"
            else:
                print(f"Login failed verification. URL: {current_url}")
                # self.save_snapshot("login_failed")
                return False, f"ログインに失敗しました（メニュー画面へ遷移できませんでした。URL: {current_url}）"
                
        except Exception as e:
            print(f"Login Loop Error: {e}")
            return False, f"ログイン処理中に例外が発生しました: {e}"

    def _calc_combinations(self, b_type, b_method, horses):
        n = len(horses)
        if b_method == '通常':
            return 1
        elif b_method == 'ボックス':
            if b_type == '単勝' or b_type == '複勝': return n
            if b_type == '枠連': return n * (n+1) // 2 
            
            import math
            def nCr(n, r): return math.comb(n, r) if n >= r else 0
            def nPr(n, r): return math.perm(n, r) if n >= r else 0
            
            if b_type == '馬連': return nCr(n, 2)
            if b_type == '馬単': return nPr(n, 2)
            if b_type == 'ワイド': return nCr(n, 2)
            if b_type == '3連複': return nCr(n, 3)
            if b_type == '3連単': return nPr(n, 3)
        elif b_method == 'フォーメーション':
            # 3連単: G1 x G2 x G3 (ただし同一馬は除外)
            # 3連複: G1 x G2 x G3 (ただし同一馬は除外、順序不同)
            # 馬単: G1 x G2
            # 馬連: G1 x G2
            # 枠連: G1 x G2
            # ワイド: G1 x G2
            
            g1 = set(horses) # API仕様次第だが、horsesに[[1,2], [3,4], [5]]のように来る想定か、
                             # あるいは bet['formation'] = {'1': [..], '2': [..]} か
            # ここでは calculate_combinations の引数 horses が list[list[int]] であると仮定する
            # もし list[int] ならばエラー
            
            # _calc_combinations は内部メソッドなので、呼び出し元で調整する。
            # horses = [[1,2], [1,2,3], [1,2,3,4]] のような構造を想定
            
            if not horses or not isinstance(horses[0], list):
                return 0
                
            if b_type == '3連単':
                count = 0
                g1 = horses[0]
                g2 = horses[1] if len(horses) > 1 else []
                g3 = horses[2] if len(horses) > 2 else []
                for i in g1:
                    for j in g2:
                        if i == j: continue
                        for k in g3:
                            if k == i or k == j: continue
                            count += 1
                return count
            
            elif b_type == '3連複':
                # 3連複フォーメーション (a-b-c)
                # 組み合わせ (set) としてユニークなもの
                g1 = horses[0]
                g2 = horses[1] if len(horses) > 1 else []
                g3 = horses[2] if len(horses) > 2 else []
                combos = set()
                for i in g1:
                    for j in g2:
                        if i == j: continue
                        for k in g3:
                            if k == i or k == j: continue
                            # sort tuple to deduplicate order
                            t = tuple(sorted([i, j, k]))
                            combos.add(t)
                return len(combos)
                
            elif b_type in ['馬単', '馬連', '枠連', 'ワイド']:
                g1 = horses[0]
                g2 = horses[1] if len(horses) > 1 else []
                count = 0
                combos = set()
                for i in g1:
                    for j in g2:
                        if i == j: continue
                        if b_type in ['馬連', '枠連', 'ワイド']:
                            t = tuple(sorted([i, j]))
                            if t not in combos:
                                combos.add(t)
                                count += 1
                        else: # 馬単
                            count += 1
                return count

        elif b_method == '流し':
            # 流し (Nagashi/Wheel) ベット
            # horses = {'axis': [軸馬番号], 'partners': [相手馬番号リスト]}
            # または horses = [[軸], [相手1, 相手2, ...]]
            
            if isinstance(horses, dict):
                axis = horses.get('axis', [])
                partners = horses.get('partners', [])
            elif isinstance(horses, list) and len(horses) >= 2:
                axis = horses[0] if isinstance(horses[0], list) else [horses[0]]
                partners = horses[1] if isinstance(horses[1], list) else horses[1:]
            else:
                return 0
            
            import math
            def nCr(n, r): return math.comb(n, r) if n >= r else 0
            
            n_axis = len(axis)
            n_partners = len(partners)
            
            if b_type == '3連複':
                # 軸1頭流し: 軸1頭 + 相手n頭から2頭 = nC2
                # 軸2頭流し: 軸2頭 + 相手n頭から1頭 = n
                if n_axis == 1:
                    return nCr(n_partners, 2)
                elif n_axis == 2:
                    return n_partners
                    
            elif b_type == '3連単':
                # 軸1頭1着固定流し: 相手n頭から2頭の順列 = nP2
                # 軸1頭2着固定などもある
                # ここでは軸1頭1着固定を想定
                if n_axis == 1:
                    return n_partners * (n_partners - 1)  # nP2
                    
            elif b_type in ['馬連', '馬単', 'ワイド']:
                # 軸流し: 軸 x 相手 = n_partners
                return n_partners

        return 0

    def _handle_important_notice(self):
        """
        「重要なお知らせ」ダイアログを処理する
        - 「今後はこのお知らせを表示しない」にチェック
        - 「OK」をクリック
        """
        try:
            # URLに 'announce' が含まれている場合、お知らせ画面にいる可能性が高い
            if 'announce' in self.driver.current_url or '重要なお知らせ' in self.driver.page_source:
                print("Important Notice detected. Handling...")
                
                # チェックボックスを探す
                # "今後はこのお知らせを表示しない" のテキストを持つlabel/span、またはその近くのcheckbox
                checkbox_candidates = [
                    "//input[@type='checkbox']",  # Generic checkbox
                    "//label[contains(text(), '今後はこのお知らせを表示しない')]//input",
                    "//label[contains(text(), '今後はこのお知らせを表示しない')]/preceding-sibling::input",
                    "//label[contains(text(), '今後はこのお知らせを表示しない')]/following-sibling::input",
                ]
                
                checkbox_found = False
                for xpath in checkbox_candidates:
                    try:
                        checkboxes = self.driver.find_elements(By.XPATH, xpath)
                        for cb in checkboxes:
                            if cb.is_displayed() and not cb.is_selected():
                                cb.click()
                                print("Checked 'Don't show again' checkbox.")
                                checkbox_found = True
                                break
                        if checkbox_found:
                            break
                    except:
                        pass
                
                if not checkbox_found:
                    # Try label click (might toggle checkbox)
                    try:
                        label = self.driver.find_element(By.XPATH, "//*[contains(text(), '今後はこのお知らせを表示しない')]")
                        label.click()
                        print("Clicked label for 'Don't show again'.")
                    except:
                        print("Could not find 'Don't show again' checkbox/label.")
                
                time.sleep(0.3)
                
                # OKボタンをクリック
                ok_btn_candidates = [
                    "//button[text()='OK']",
                    "//input[@value='OK']",
                    "//a[text()='OK']",
                    "//button[contains(text(), 'OK')]",
                    "//*[text()='OK']",
                ]
                
                ok_clicked = False
                for xpath in ok_btn_candidates:
                    try:
                        btn = self.driver.find_element(By.XPATH, xpath)
                        if btn.is_displayed():
                            btn.click()
                            print("Clicked 'OK' button on Important Notice.")
                            ok_clicked = True
                            break
                    except:
                        pass
                
                if not ok_clicked:
                    print("Could not find 'OK' button on Important Notice.")
                    
                time.sleep(self.WAIT_SEC_LONG)
                
        except Exception as e:
            print(f"Error handling Important Notice: {e}")

    def vote(self, race_id: str, bets: List[Dict[str, Any]], stop_at_confirmation: bool = True) -> tuple[bool, str]:
        """
        投票を実行する
        """
        if not self.driver:
            return False, "Browser not initialized."
            
        try:
            # 0. 重要なお知らせダイアログを処理
            self._handle_important_notice()
            
            # 1. 開催情報の解析
            # race_idから場所、レース番号などを特定
            place_code = race_id[4:6] # 05 -> 東京
            race_num = int(race_id[10:12]) # 11 -> 11R
            
            # 場所コードマッピング (netkeiba -> IPAT PC)
            # IPATの場所コードは内部的だが、画面上は「東京」「中山」などのテキストで選択
            place_name_map = {
                '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
                '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
            }
            target_place_name = place_name_map.get(place_code, "東京") # Default to Tokyo
            
            print(f"Target: {target_place_name} {race_num}R")
            
            # 2. トップメニューから「通常投票」を選択
            # ユーザー情報: 通常投票ボタン -> ポップアップ -> レース選択
            
            # 2. トップメニューから「通常投票」を選択
            # SPA (AngularJS) なので button タグを探す
            try:
                print("Searching for 'Normal Vote' button (SPA)...")
                # <button ui-sref="bet.basic" ...>
                normal_vote_selector = "//button[@ui-sref='bet.basic']"
                
                # 待機してからクリック
                btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, normal_vote_selector))
                )
                btn.click()
                print("Clicked 'Normal Vote' button.")
                time.sleep(self.WAIT_SEC_LONG)
                
            except Exception as e:
                print(f"Normal Vote button navigation failed: {e}")
                self._save_debug_screenshot(self.driver, "normal_vote_fail")
                # 続行不能ならリターンすべきだが、手動リカバリの可能性を残す

            # 3. ポップアップ画面の処理 ("このまま進む")
            # SPAのモーダルダイアログ。レンダリング待ちが必要。
            try:
                print("Checking for popup...")
                # 少し待ってからチェック
                try:
                    # 3秒待ってみる（ポップアップアニメーション等）
                    popup_btn = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'このまま進む')]"))
                    )
                    popup_btn.click()
                    print("Popup handled ('Proceed' clicked).")
                    time.sleep(self.WAIT_SEC_LONG)
                except:
                    print("No popup found (or timed out).")
            except Exception as e:
                print(f"Popup check error: {e}")

            # 4. 場所・レース番号選択 (SPA画面)
            try:
                print(f"Selecting Place: {target_place_name} Race: {race_num}R")
                
                found_place = False
                
                # ケース1: 既に投票画面（プルダウンがある）
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for sel in select_elements:
                    try:
                        options = sel.find_elements(By.TAG_NAME, "option")
                        for opt in options:
                            if target_place_name in opt.text:
                                Select(sel).select_by_visible_text(opt.text)
                                print(f"Selected place '{opt.text}' via dropdown.")
                                found_place = True
                                time.sleep(self.WAIT_SEC)
                                break
                    except: pass
                    if found_place: break
                
                if not found_place:
                    # ケース2: 場名選択画面（ボタン）
                    print("Dropdown not found, searching for place buttons...")
                    # <button>...<span>東京（日）</span>...</button>
                    # テキスト全体検索 (.) を使う
                    xpath = f"//button[contains(., '{target_place_name}')]"
                    place_elems = self.driver.find_elements(By.XPATH, xpath)
                    for el in place_elems:
                        if el.is_displayed():
                            el.click()
                            print(f"Clicked place button: {target_place_name}")
                            found_place = True
                            time.sleep(self.WAIT_SEC) # レース一覧更新待ち
                            break
                            
                    # フォールバック: divも探す
                    if not found_place:
                        xpath_div = f"//div[contains(., '{target_place_name}')]"
                        divs = self.driver.find_elements(By.XPATH, xpath_div)
                        for d in divs:
                             # クリック可能なdivか判断するのは難しいが、classやroleを見る
                             # とりあえずクリックトライ
                             if d.is_displayed() and ("btn" in d.get_attribute("class") or "tab" in d.get_attribute("class")):
                                 d.click()
                                 print(f"Clicked place div: {target_place_name}")
                                 found_place = True
                                 time.sleep(self.WAIT_SEC)
                                 break
                
                # レース選択 (1R, 2R...)
                print(f"Selecting Race {race_num}R...")
                race_done = False
                
                # プルダウン再検索
                select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                for sel in select_elements:
                    try:
                        if f"{race_num}R" in sel.text: 
                            race_select = Select(sel)
                            for opt in race_select.options:
                                if f"{race_num}R" in opt.text:
                                    race_select.select_by_visible_text(opt.text)
                                    print(f"Selected race '{opt.text}' via dropdown.")
                                    race_done = True
                                    time.sleep(self.WAIT_SEC)
                                    break
                    except: pass
                    if race_done: break
                
                if not race_done:
                    # ボタン検索 (SPA構造対応)
                    # <button><div class="race-no"><span>11</span>R</div>...</button>
                    # {race_num} と完全一致する span を持ち、その親(または先祖)のbuttonをクリック
                    xpath_race = f"//span[normalize-space(text())='{race_num}']/ancestor::button"
                    btns = self.driver.find_elements(By.XPATH, xpath_race)
                    for btn in btns:
                        if btn.is_displayed():
                            # click
                            btn.click()
                            print(f"Clicked race button for {race_num}R")
                            race_done = True
                            time.sleep(self.WAIT_SEC_LONG)
                            break
                    
                    if not race_done:
                         # 以前のロジック: テキスト検索 (contains)
                         xpath_fallback = f"//button[contains(., '{race_num}R')]"
                         btns = self.driver.find_elements(By.XPATH, xpath_fallback)
                         for btn in btns:
                             if btn.is_displayed():
                                 btn.click()
                                 print(f"Clicked race button (fallback) for {race_num}R")
                                 race_done = True
                if not race_done:
                    print("WARNING: Could not auto-select race.")
                    # DEBUG: Save source on failure
                    # debug_fail_path = os.path.join(os.getcwd(), "debug_race_selection_fail.html")
                    # with open(debug_fail_path, "w", encoding="utf-8") as f:
                    #     f.write(self.driver.page_source)
                    # print(f"DEBUG: Saved failure source to {debug_fail_path}")

            except Exception as e:
                print(f"Place/Race selection warning: {e}")
                import traceback
                traceback.print_exc()
                
                # JSでクリックを試みる (高速かつ確実)
                script_race_click = f"""
                var targets = ['{race_num}R', '{race_num} R', '{race_num}'];
                var links = document.getElementsByTagName('a');
                for(var i=0; i<links.length; i++){{
                    var text = links[i].innerText.trim();
                    if(targets.includes(text)){{
                        links[i].click();
                        return true;
                    }}
                    // 画像ボタンの場合 alt属性チェック
                    var imgs = links[i].getElementsByTagName('img');
                    if(imgs.length > 0){{
                        var alt = imgs[0].alt;
                        if(targets.includes(alt)){{
                            links[i].click();
                            return true;
                        }}
                    }}
                }}
                return false;
                """
                result = self.driver.execute_script(script_race_click)
                
                if not result:
                    # Selenium Fallback
                    race_xpath = f"//a[contains(text(), '{race_num}R') or text()='{race_num}']"
                    race_btn = self.driver.find_element(By.XPATH, race_xpath)
                    race_btn.click()
                
                time.sleep(self.WAIT_SEC)
            except Exception as e:
                print(f"Race selection error: {e}")
                import traceback
                traceback.print_exc()
                # ここで諦めず、ユーザーに助けを求める
                print(f">>> Automatic selection of {race_num}R failed.")
                print(">>> Please SELECT THE RACE MANUALLY in the browser.")
                print(">>> Then press Enter here to continue...")
                input()
                # 画面遷移していない可能性が高いが、続行不可
                return False, f"レース番号({race_num}R)の選択に失敗しました: {e}"

            # DEBUG: Save voting screen source
            try:
                # レース選択後、少し待ってから保存
                time.sleep(2.0)
                debug_vote_path = os.path.join(os.getcwd(), "debug_voting_screen.html")
                with open(debug_vote_path, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"DEBUG: Saved voting screen source to {debug_vote_path}")
            except: pass

            # 4. 投票入力ループ (高速化版)
            # waitを極力排除し、JSで直接DOMを操作する
            
            entered_count = 0
            
            for bet in bets:
                time.sleep(0.1) # Speedup from 2.0
                try:
                    b_type = bet['type']      # 単勝, 複勝, etc
                    b_method = bet.get('method', '通常') # 通常, ボックス
                    b_horses = bet['horses']  # List[int]
                    b_amount = bet.get('amount', 100)
                    
                    print(f"Entering Bet: Type={b_type}, Method={b_method}, Horses={b_horses}, Amount={b_amount}")
                    
                    # 1. 式別選択 (Pull-down)
                    # マッピング
                    type_map = {
                        "馬連複": "馬連",
                        "馬連単": "馬単", 
                        "3連複": "３連複",
                        "3連単": "３連単",
                        "ワイド": "ワイド",
                        "単勝": "単勝",
                        "複勝": "複勝",
                        "枠連": "枠連"
                    }
                    target_type_name = type_map.get(b_type, b_type)
                    
                    try:
                        type_select_elem = self.driver.find_element(By.ID, "bet-basic-type")
                        Select(type_select_elem).select_by_visible_text(target_type_name)
                        # Angular Update Force
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", type_select_elem)
                        time.sleep(0.1) # Speedup from 0.5
                    except Exception as ex:
                        print(f"Failed to select Bet Type '{target_type_name}': {ex}")
                    
                    # 2. 方式選択 (通常/ボックス)
                    try:
                        method_selected = False
                        
                        # Strategy A: Try to find a <select> element for Method (Way)
                        selects = self.driver.find_elements(By.TAG_NAME, "select")
                        for sel in selects:
                            try:
                                if sel.get_attribute("id") == "bet-basic-type":
                                    continue
                                options = sel.find_elements(By.TAG_NAME, "option")
                                for opt in options:
                                    if b_method == opt.text.strip():
                                        Select(sel).select_by_visible_text(b_method)
                                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", sel)
                                        print(f"Selected Method '{b_method}' via Dropdown (Select).")
                                        method_selected = True
                                        time.sleep(0.1) # Speedup from 0.5
                                        break
                                if method_selected: break
                            except: pass
                        
                        # Strategy B: Click Buttons/Tabs/Links
                        if not method_selected:
                            xpath_method = f"//*[contains(text(), '{b_method}')]"
                            method_elems = self.driver.find_elements(By.XPATH, xpath_method)
                            
                            for el in method_elems:
                                if el.is_displayed():
                                    tagName = el.tag_name.lower()
                                    if tagName in ['a', 'button', 'label', 'span', 'li', 'div']:
                                        try:
                                            if tagName == 'option': continue
                                            el.click()
                                            method_selected = True
                                            print(f"Selected Method '{b_method}' via Click ({tagName}).")
                                            time.sleep(0.1) # Speedup from 0.5
                                            break
                                        except: pass
                        
                        if not method_selected and b_method != '通常':
                            print(f"Warning: Method '{b_method}' not found.")
                            
                        # DEBUG: Screenshot after method selection
                        self._save_debug_screenshot(self.driver, f"after_method_{b_type}_{b_method}")

                    except Exception as e:
                        print(f"Method selection error: {e}")

                    # 3. 馬番選択 (Label Click)
                    if b_method == 'フォーメーション':
                        print(f"Formation Selection Mode: {b_horses}")
                        if not isinstance(b_horses, list) or not isinstance(b_horses[0], list):
                            print("Error: For formation, 'horses' must be list of lists.")
                            continue
                            
                        ranks = ["1着", "2着", "3着"] 
                        if b_type in ['馬連', '馬単', '枠連', 'ワイド']:
                             ranks = ["1頭目", "2頭目"]
                        
                        for i, rank_horses in enumerate(b_horses):
                            if i >= len(ranks): break
                            target_rank_name = ranks[i]
                            print(f"selecting rank {i+1} ({target_rank_name}): {rank_horses}")
                            
                            for horse_num in rank_horses:
                                try:
                                    xpath_no = f"//span[contains(@class, 'ipat-racer-no') and normalize-space(text())='{horse_num}']/ancestor::tr"
                                    row = self.driver.find_element(By.XPATH, xpath_no)
                                    inputs = row.find_elements(By.TAG_NAME, "input")
                                    labels = row.find_elements(By.TAG_NAME, "label")
                                    valid_inputs = [inp for inp in inputs if inp.get_attribute("type") == "checkbox" or inp.get_attribute("type") == "radio"]
                                    
                                    target_input = None
                                    target_label = None
                                    if len(valid_inputs) > i:
                                        target_input = valid_inputs[i]
                                        inp_id = target_input.get_attribute("id")
                                        if inp_id:
                                            for lbl in labels:
                                                if lbl.get_attribute("for") == inp_id:
                                                    target_label = lbl
                                                    break
                                    
                                    if target_label:
                                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_label)
                                        target_label.click()
                                    elif target_input:
                                        self.driver.execute_script("arguments[0].click();", target_input)
                                    else:
                                        print(f"Could not find input for Horse {horse_num} Rank {i+1}")
                                except Exception as e:
                                    print(f"Error selecting formation horse {horse_num} rank {i+1}: {e}")

                    elif b_method == '流し':
                        # 流し (Nagashi/Wheel) 選択モード
                        print(f"Nagashi Selection Mode: {b_horses}")
                        
                        # horses構造: {'axis': [軸], 'partners': [相手]} または [[軸], [相手]]
                        if isinstance(b_horses, dict):
                            axis_horses = b_horses.get('axis', [])
                            partner_horses = b_horses.get('partners', [])
                        elif isinstance(b_horses, list) and len(b_horses) >= 2:
                            axis_horses = b_horses[0] if isinstance(b_horses[0], list) else [b_horses[0]]
                            partner_horses = b_horses[1] if isinstance(b_horses[1], list) else b_horses[1:]
                        else:
                            print("Error: Invalid Nagashi horses structure.")
                            continue
                        
                        # IPAT流し画面: 「軸」列と「相手」列がある
                        # 軸馬は軸列のチェックボックス (index 0)
                        # 相手馬は相手列のチェックボックス (index 1)
                        
                        print(f"Axis horses: {axis_horses}")
                        print(f"Partner horses: {partner_horses}")
                        
                        # 軸馬選択 (0番目のチェックボックス)
                        for horse_num in axis_horses:
                            try:
                                xpath_no = f"//span[contains(@class, 'ipat-racer-no') and normalize-space(text())='{horse_num}']/ancestor::tr"
                                row = self.driver.find_element(By.XPATH, xpath_no)
                                inputs = row.find_elements(By.TAG_NAME, "input")
                                labels = row.find_elements(By.TAG_NAME, "label")
                                valid_inputs = [inp for inp in inputs if inp.get_attribute("type") in ("checkbox", "radio")]
                                
                                if valid_inputs:
                                    target_input = valid_inputs[0]  # 軸列 (1st column)
                                    inp_id = target_input.get_attribute("id")
                                    target_label = None
                                    if inp_id:
                                        for lbl in labels:
                                            if lbl.get_attribute("for") == inp_id:
                                                target_label = lbl
                                                break
                                    
                                    if target_label:
                                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_label)
                                        target_label.click()
                                        print(f"Selected AXIS horse: {horse_num}")
                                    elif target_input:
                                        self.driver.execute_script("arguments[0].click();", target_input)
                                        print(f"Selected AXIS horse: {horse_num} (JS)")
                            except Exception as e:
                                print(f"Error selecting axis horse {horse_num}: {e}")
                        
                        # 相手馬選択 (1番目のチェックボックス)
                        for horse_num in partner_horses:
                            try:
                                xpath_no = f"//span[contains(@class, 'ipat-racer-no') and normalize-space(text())='{horse_num}']/ancestor::tr"
                                row = self.driver.find_element(By.XPATH, xpath_no)
                                inputs = row.find_elements(By.TAG_NAME, "input")
                                labels = row.find_elements(By.TAG_NAME, "label")
                                valid_inputs = [inp for inp in inputs if inp.get_attribute("type") in ("checkbox", "radio")]
                                
                                if len(valid_inputs) > 1:
                                    target_input = valid_inputs[1]  # 相手列 (2nd column)
                                    inp_id = target_input.get_attribute("id")
                                    target_label = None
                                    if inp_id:
                                        for lbl in labels:
                                            if lbl.get_attribute("for") == inp_id:
                                                target_label = lbl
                                                break
                                    
                                    if target_label:
                                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_label)
                                        target_label.click()
                                        print(f"Selected PARTNER horse: {horse_num}")
                                    elif target_input:
                                        self.driver.execute_script("arguments[0].click();", target_input)
                                        print(f"Selected PARTNER horse: {horse_num} (JS)")
                                else:
                                    print(f"Warning: Not enough input columns for partner horse {horse_num}")
                            except Exception as e:
                                print(f"Error selecting partner horse {horse_num}: {e}")

                    else:
                        # 通常・BOX (単一リスト)
                        try:
                            try:
                                WebDriverWait(self.driver, 10).until(
                                    EC.presence_of_element_located((By.XPATH, "//span[contains(@class, 'ipat-racer-no')]"))
                                )
                            except:
                                print("Timeout waiting for ANY horse label")

                            for horse_num in b_horses:
                                label_for = f"no{horse_num}"
                                try:
                                    label_xpath = f"//label[@for='{label_for}']"
                                    labels = self.driver.find_elements(By.XPATH, label_xpath)
                                    
                                    target_label = None
                                    if labels:
                                        target_label = labels[0]
                                    else:
                                        print(f"Label ID search failed for {horse_num}, trying Text search...")
                                        xpath_text = f"//span[contains(@class, 'ipat-racer-no') and normalize-space(text())='{horse_num}']/ancestor::tr//label"
                                        labels_text = self.driver.find_elements(By.XPATH, xpath_text)
                                        if labels_text:
                                            target_label = labels_text[0]
                                    
                                    if target_label:
                                        input_id = target_label.get_attribute("for")
                                        if not input_id:
                                             input_id = label_for
                                        
                                        try:
                                            input_elem = self.driver.find_element(By.ID, input_id)
                                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_elem)
                                            time.sleep(0.1)
    
                                            if not input_elem.is_selected():
                                                try:
                                                    target_label.click()
                                                except Exception as ck_ex:
                                                    print(f"Label click failed/intercepted ({ck_ex}), trying JS Click on Label...")
                                                    try:
                                                        self.driver.execute_script("arguments[0].click();", target_label)
                                                    except: pass
    
                                                if not input_elem.is_selected():
                                                    print(f"Click failed for {horse_num}, trying JS Click on Input...")
                                                    self.driver.execute_script("arguments[0].click();", input_elem)
                                                    time.sleep(0.5)
                                                
                                                if input_elem.is_selected():
                                                    print(f"Selected Horse: {horse_num} (Verified)")
                                                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", input_elem)
                                                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", input_elem)
                                                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('click'));", input_elem)
                                                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('blur'));", input_elem)
                                                else:
                                                    print(f"FAILED to select Horse: {horse_num}")
                                            else:
                                                print(f"Horse {horse_num} already selected.")
                                        except Exception as ex:
                                            print(f"Input element check failed ({ex}), clicking label blindly.")
                                            try:
                                                self.driver.execute_script("arguments[0].click();", target_label)
                                                print(f"Selected Horse: {horse_num} (Blind JS)")
                                            except Exception as bex:
                                                print(f"Blind click failed: {bex}")
                                            time.sleep(0.5)
                                    else:
                                         print(f"Horse Label not found for: {horse_num}")
    
                                except Exception as ex:
                                    print(f"Error selecting horse {horse_num}: {ex}")
                            
                            print("Waiting for combination count update (nTotalNum > 0)...")
                            try:
                                WebDriverWait(self.driver, 5).until(
                                    lambda d: d.execute_script("""
                                        var el = document.querySelector('input[ng-model="vm.nUnit"]');
                                        if(el) {
                                            var scope = angular.element(el).scope();
                                            return scope && scope.vm && scope.vm.nTotalNum > 0;
                                        }
                                        return false;
                                    """)
                                )
                                print("Combination count updated (Verified via JS).")
                            except:
                                print("Combination count wait timed out (nTotalNum stayed 0).")
                                
                            print("Waiting for Amount Input to become enabled...")
                            try:
                                amount_chk = self.driver.find_element(By.CSS_SELECTOR, "input[ng-model='vm.nUnit']")
                                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", amount_chk)
                                try:
                                    WebDriverWait(self.driver, 5).until(lambda d: amount_chk.is_enabled())
                                    print("Amount input verified enabled.")
                                except:
                                    print(f"Amount input still disabled (Attr: {amount_chk.get_attribute('disabled')})")
                                    pass
                            except Exception as e:
                                print(f"Error checking amount input: {e}")
    
                        except Exception as e:
                            print(f"Horse selection error: {e}")
                    
                    # 4. 金額入力 (Input)
                    try:
                        amount_input = self.driver.find_element(By.CSS_SELECTOR, "input[ng-model='vm.nUnit']")
                        
                        # 明示的に待機
                        print("Waiting for Amount Input to become enabled...")
                        try:
                            WebDriverWait(self.driver, 5).until(lambda d: amount_input.is_enabled())
                        except:
                            print("Amount input is still disabled after wait!")
                        
                        if not amount_input.is_enabled():
                            print("Amount input is disabled! Horse selection might have failed.")
                            self._save_debug_screenshot(self.driver, f"amount_disabled_{b_type}_{b_method}")
                        
                        # クリアして入力
                        amount_input.clear()
                        # 1=100円の単位
                        input_val = str(b_amount // 100)
                        
                        amount_input.send_keys(input_val)
                        print(f"Entered Amount: {input_val} (x100 yen)")
                        amount_input.send_keys(Keys.TAB)
                        
                        # DEBUG: Read Angular Scope variables AFTER Input
                        try:
                            scope_debug_post = self.driver.execute_script("""
                                var el = document.querySelector('input[ng-model="vm.nUnit"]');
                                if(el) {
                                    var scope = angular.element(el).scope();
                                    if(scope && scope.vm) {
                                        return "nTotalNum=" + scope.vm.nTotalNum + ", nUnit=" + scope.vm.nUnit + ", bLoading=" + scope.vm.bLoading;
                                    }
                                }
                                return "Scope not found";
                            """)
                            print(f"DEBUG ANGULAR SCOPE (Post-Input): {scope_debug_post}")
                            with open("debug_angular_scope.txt", "a", encoding="utf-8") as f:
                                f.write(f"{datetime.datetime.now()} [Post-Input]: {scope_debug_post}\n")
                        except Exception as dbg_ex:
                            print(f"Scope debug post-input failed: {dbg_ex}")

                        # Angularのモデル更新のためイベント発火
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", amount_input)
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", amount_input)
                        self.driver.execute_script("arguments[0].dispatchEvent(new Event('blur'));", amount_input)
                        
                        # time.sleep(1.0) # Removed, Set button check below handles it
                        
                    except Exception as e:
                        print(f"Amount input error: {e}")
                    

                    # 5. セットボタン (Set)
                    # User provided: <button ... ng-click="vm.onSet()" ...>セット</button>
                    try:
                        # Wait for button to be clickable (it becomes enabled after amount input)
                        # Switch to ng-click selector as per user snippet
                        set_btn_xpath = "//button[contains(@ng-click, 'onSet')]"
                        try:
                            # Debug: Check how many set buttons exist
                            set_btns = self.driver.find_elements(By.XPATH, set_btn_xpath)
                            # print(f"Found {len(set_btns)} Set buttons.")

                            set_btn = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, set_btn_xpath))
                            )
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", set_btn)
                            # time.sleep(0.5) # Removed
                            
                            # Click
                            print(f"Clicking Set button...")
                            try:
                                set_btn.click()
                            except:
                                self.driver.execute_script("arguments[0].click();", set_btn)
                            
                            clicked = True
                            print("Clicked 'Set' button.")
                        except Exception as ex:
                            print(f"Exact 'Set' button issue: {ex}")
                            clicked = False
                            
                            # Fallback logic removed/simplified as we trust the specific selector now
                            # But keep last resort search for "セット" text if ng-click fails?
                            if not clicked:
                                print("Trying fallback search for 'セット' text...")
                                try:
                                    fb_btn = self.driver.find_element(By.XPATH, "//button[normalize-space(text())='セット']")
                                    if fb_btn.is_displayed():
                                         fb_btn.click()
                                         clicked = True
                                         print("Clicked Set button (Fallback Text).")
                                except: pass

                            if not clicked:
                                print("Failed to click any Set button.")
                                # DEBUG: Print all button texts
                                all_btns = self.driver.find_elements(By.TAG_NAME, "button")
                                btn_texts = [b.text.strip().replace('\n', '') for b in all_btns if b.is_displayed()]
                                print(f"Visible buttons: {btn_texts}")
                                
                                self._save_debug_screenshot(self.driver, f"set_button_fail_{b_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                                # Save HTML too
                                with open(f"debug_set_fail_{b_type}.html", "w", encoding="utf-8") as f:
                                    f.write(self.driver.page_source)

                        entered_count += 1
                        time.sleep(0.2) # Keeping small buffer for potential list update, but reduced to 0.2

                        
                        # Check if error occurred (e.g. popups)
                        try:
                             alert = self.driver.switch_to.alert
                             print(f"Alert found after Set: {alert.text}")
                             alert.accept()
                        except: pass
                        
                        # Check for error messages on page
                        try:
                             err_msg = self.driver.find_elements(By.CLASS_NAME, "list-error")
                             for em in err_msg:
                                 if em.is_displayed():
                                     print(f"Error message displayed: {em.text}")
                        except: pass

                    except Exception as e:
                        print(f"Set button error: {e}")
                    
                        
                except Exception as e:
                    print(f"Bet processing error: {e}")
                    continue
            
            if entered_count == 0:
                print("No bets entered.")
                return False, "投票データの入力に失敗しました"
                
            # 6. 購入予定リストへの遷移 (Cart)
            try:
                # <button class="btn btn-vote-list" ...>
                cart_btn = self.driver.find_element(By.CSS_SELECTOR, "button.btn-vote-list")
                cart_btn.click()
                print("Opened Bet List.")
                time.sleep(self.WAIT_SEC_LONG)
            except Exception as e:
                return False, f"購入予定リストへの遷移に失敗: {e}"
                
            # 7. 合計金額入力 (Total)
            try:
                # Calculate Expected Total
                total_yen = 0
                for b in bets:
                    try:
                        b_type = b.get('type')
                        b_method = b.get('method', '通常')
                        b_horses = b.get('horses', [])
                        unit_yen = b.get('amount', 100)
                        
                        combos = self._calc_combinations(b_type, b_method, b_horses)
                        cost = combos * unit_yen
                        total_yen += cost
                    except: pass
                
                print(f"Calculated Total Amount: {total_yen} Yen")

                # Wait/Find Total Input
                # <input ... ng-model="vm.cAmountTotal">
                total_input = WebDriverWait(self.driver, 5).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "input[ng-model='vm.cAmountTotal']"))
                )
                
                total_input.clear()
                total_input.send_keys(str(total_yen))
                total_input.send_keys(Keys.TAB)
                
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", total_input)
                self.driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", total_input)
                
                print(f"Entered Total Amount: {total_yen}")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Total amount input warning: {e}")
                
            # 8. 確認 (Verification Stop)
            if stop_at_confirmation:
                # 購入ボタン手前で止める
                self._save_snapshot("stopped_at_confirmation_spa")
                return True, "確認画面で停止しました。内容を確認して投票ボタンを押してください。"
            
            # (自動投票する場合のロジックはここに追加)
            # 購入ボタン: button[ng-click="vm.clickPurchase()"]
            return True, "完了(確認待ち)"

        except Exception as e:
            self._save_debug_screenshot(self.driver, "vote_error")
            return False, f"PC版投票エラー: {e}"

    def close(self):
        if self.driver:
            self.driver.quit()
        
    def _save_snapshot(self, name):
         """Save screenshot and page source for debugging."""
         if not self.driver or not self.debug_mode: return
         
         try:
             timestamp = datetime.datetime.now().strftime("%H%M%S")
             filename_base = f"debug_{timestamp}_{name}"
             self.driver.save_screenshot(f"{filename_base}.png")
             with open(f"{filename_base}.html", "w", encoding="utf-8") as f:
                 f.write(self.driver.page_source)
         except:
             pass
