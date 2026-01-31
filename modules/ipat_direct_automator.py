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
            
            # PC版ログインフォーム入力
            # 加入者番号: name="i"
            # P-ARS番号: name="r"
            # INET-ID: name="inetid"
            # ログインボタン: onclick="DoLogin()" or class="login_btn"
            
            try:
                # 1. 加入者番号
                self.driver.find_element(By.NAME, "i").send_keys(subscriber_no)
                time.sleep(0.1)
                
                # 2. P-ARS番号
                self.driver.find_element(By.NAME, "r").send_keys(pars_no)
                time.sleep(0.1)
                
                # 3. INET-ID
                self.driver.find_element(By.NAME, "inetid").send_keys(inetid)
                time.sleep(0.1)
                
                # 4. ログインボタン
                # PC版は通常CSSセレクタかXPathでクリック
                # 構造が変わる可能性があるため、いくつか候補を試す
                login_btn_candidates = [
                    "//a[contains(@onclick, 'DoLogin')]",
                    "//div[contains(@class, 'login_btn')]//a",
                    "//input[@type='image' and contains(@alt, 'ログイン')]"
                ]
                
                clicked = False
                for xpath in login_btn_candidates:
                    try:
                        btn = self.driver.find_element(By.XPATH, xpath)
                        btn.click()
                        clicked = True
                        break
                    except:
                        continue
                
                if not clicked:
                     # 最後の手段: JS実行
                     self.driver.execute_script("DoLogin();")
                
                time.sleep(self.WAIT_SEC_LONG)
                
                # 5. 暗証番号入力 (画面遷移後)
                # PC版は加入者情報入力 -> 暗証番号入力 の2段階認証の場合がある
                # あるいは1画面で済む場合もある。画面遷移を見て判断
                
                # 暗証番号入力要素を探す
                # name="p" が一般的
                try:
                    pin_input = self.driver.find_element(By.NAME, "p")
                    pin_input.send_keys(pin)
                    
                    # 認証ボタン
                    # onclick="DoAuth()" ?
                    auth_btn_candidates = [
                        "//a[contains(@onclick, 'DoAuth')]",
                        "//a[contains(text(), '確認')]",
                        "//a[contains(text(), 'ログイン')]" # 2段階目のボタンもログイン表記の場合あり
                    ]
                    
                    clicked_auth = False
                    for xpath in auth_btn_candidates:
                        try:
                            btn = self.driver.find_element(By.XPATH, xpath)
                            btn.click()
                            clicked_auth = True
                            break
                        except:
                            continue
                            
                    if not clicked_auth:
                        self.driver.execute_script("DoAuth();")
                        
                    time.sleep(self.WAIT_SEC_LONG)
                    
                except:
                    print("PIN input skipped or merged in previous screen.")
                
                # ログイン成功確認
                # メニュー画面等が表示されているか
                time.sleep(1.0)
                if "ネット投票" in self.driver.page_source or "投票メニュー" in self.driver.page_source:
                    print("Login success.")
                    return True, "ログイン成功"
                else:
                    # エラーメッセージ取得
                    err_msg = "Login verification failed."
                    try:
                        err_el = self.driver.find_element(By.CSS_SELECTOR, ".error, .err, .msg")
                        err_msg = err_el.text
                    except:
                        pass
                    return False, f"ログイン失敗: {err_msg}"
                    
            except Exception as e:
                print(f"Login process error: {e}")
                self._save_debug_screenshot(self.driver, "login_error")
                return False, f"ログイン処理エラー: {e}"
                
        except Exception as e:
            return False, f"ブラウザ起動/アクセスエラー: {e}"

    def vote(self, race_id: str, bets: List[Dict[str, Any]], stop_at_confirmation: bool = True) -> tuple[bool, str]:
        """
        投票を実行する
        
        Args:
            race_id: レースID (netkeiba形式: 202401010101)
                     YYYY + BB + KK + DD + RR
                     BB: 場所コード (01:札幌, 05:東京...)
                     KK: 回次
                     DD: 日次
                     RR: レース番号
            bets: 投票データのリスト
            stop_at_confirmation: 確認画面で停止するかどうか
        """
        if not self.driver:
            return False, "Browser not initialized."
            
        try:
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
            
            # 2. トップメニューから「ネット投票」を選択
            # PC版トップメニュー
            try:
                # "ネット投票" ボタンを探す
                # 通常は大きなボタン
                net_vote_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), 'ネット投票') or contains(@title, 'ネット投票')]")
                net_vote_btn.click()
                time.sleep(self.WAIT_SEC_LONG)
            except Exception as e:
                print(f"Menu navigation error: {e}")
                # 既に投票画面にいる可能性もあるので続行してみる
            
            # Window Handler処理が必要な場合も考慮（ポップアップが出る場合）
            # PC版IPATはメインウィンドウ内で遷移することが多いが、確認が必要
            
            # 3. 開催日・競馬場選択
            # 曜日/開催日選択画面が出る場合がある（土日開催時など）
            # 土曜日分の前日発売など
            
            # 画面遷移: 投票メニュー -> (開催選択) -> レース選択 -> (方式選択) -> 入力
            
            # ここから先は汎用的な「通常投票」フローを想定
            
            # 場所選択
            try:
                # ボタンテキストで場所を選択
                # 例: <a href="...">東京</a>
                place_btn = self.driver.find_element(By.XPATH, f"//a[contains(text(), '{target_place_name}')]")
                place_btn.click()
                time.sleep(self.WAIT_SEC)
            except Exception as e:
                print(f"Place selection warning: {e}")
                # 自動選択されている場合や、既に選択済みの場合もある
                
            # レース番号選択
            try:
                # 11R などのボタン
                # ボタンのテキストは "11" または "11R"
                race_xpath = f"//a[contains(text(), '{race_num}R') or text()='{race_num}']"
                race_btn = self.driver.find_element(By.XPATH, race_xpath)
                race_btn.click()
                time.sleep(self.WAIT_SEC)
            except Exception as e:
                print(f"Race selection error: {e}")
                return False, f"レース番号({race_num}R)の選択に失敗しました"

            # 4. 投票入力ループ (高速化版)
            # waitを極力排除し、JSで直接DOMを操作する
            
            entered_count = 0
            
            for bet in bets:
                try:
                    b_type = bet['type']      # 単勝, 複勝, etc
                    b_method = bet.get('method', '通常') # 通常, ボックス
                    b_horses = bet['horses']  # List[int]
                    b_amount = bet.get('amount', 100)
                    
                    # 1. 式別選択 (JS click)
                    # IPAT PC版はタブ切り替えが必要
                    script_type = f"""
                    var links = document.getElementsByTagName('a');
                    for(var i=0; i<links.length; i++){{
                        if(links[i].innerText.indexOf('{b_type}') !== -1){{
                            links[i].click();
                            return true;
                        }}
                    }}
                    return false;
                    """
                    self.driver.execute_script(script_type)
                    time.sleep(0.1) # 画面切り替えの最小ウェイト
                    
                    # 2. 方式選択 (通常/ボックス)
                    # デフォルトが通常の場合が多いが、ボックスなら切り替えが必要
                    if b_method == 'ボックス':
                        script_method = """
                        var links = document.getElementsByTagName('a');
                        for(var i=0; i<links.length; i++){
                            if(links[i].innerText.indexOf('ボックス') !== -1){
                                links[i].click();
                                break;
                            }
                        }
                        """
                        self.driver.execute_script(script_method)
                        time.sleep(0.1)
                    elif b_method == '通常':
                        # 念のため通常ボタンがあれば押す
                        script_method = """
                        var links = document.getElementsByTagName('a');
                        for(var i=0; i<links.length; i++){
                            if(links[i].innerText.indexOf('通常') !== -1){
                                links[i].click();
                                break;
                            }
                        }
                        """
                        self.driver.execute_script(script_method)
                        time.sleep(0.1)

                    # 3. 馬番選択 (JS Check)
                    # 馬番はチェックボックス or ラベル
                    # 一括でチェックを入れるJS
                    try:
                        horses_str_list = [f"{h:02d}" for h in b_horses] # ['01', '09']
                        script_horses = f"""
                        var targets = {horses_str_list};
                        var inputs = document.getElementsByTagName('input');
                        var labels = document.getElementsByTagName('label');
                        
                        // Clear all checkboxes first (safety)
                        // for(var i=0; i<inputs.length; i++){{
                        //    if(inputs[i].type == 'checkbox' && inputs[i].checked) inputs[i].click();
                        // }}
                        
                        for(var t=0; t<targets.length; t++){{
                            var val = targets[t];
                            var found = false;
                            // Check by value
                            for(var i=0; i<inputs.length; i++){{
                                if(inputs[i].value == val && inputs[i].type == 'checkbox'){{
                                    if(!inputs[i].checked) inputs[i].click();
                                    found = true;
                                    break;
                                }}
                            }}
                            if(!found){{
                                // Check by label text
                                for(var i=0; i<labels.length; i++){{
                                    if(labels[i].innerText.indexOf(val) !== -1 && labels[i].className.indexOf('umaban') !== -1){{
                                        labels[i].click();
                                        break;
                                    }}
                                }}
                            }}
                        }}
                        """
                        self.driver.execute_script(script_horses)
                    except Exception as e:
                        print(f"JS Horse selection error: {e}")
                    
                    # 4. 金額入力 (JS Value Set)
                    try:
                        # name="money" or similar
                        # 100円単位で入力することが多い: 100円 -> "1"
                        # しかしIPATの新UIでは直接金額の場合もある。
                        # 安全策: 要素を探して標準入力
                        input_val = str(b_amount // 100) # 単位: 100円
                        script_money = f"""
                        var moneyParams = document.getElementsByName('money');
                        if(moneyParams.length > 0){{
                            moneyParams[0].value = '{input_val}';
                            // Trigger change event just in case
                            var event = new Event('change');
                            moneyParams[0].dispatchEvent(event);
                        }}
                        """
                        self.driver.execute_script(script_money)
                    except:
                        pass
                        
                    # 5. セットボタン (JS Click)
                    script_set = """
                    var links = document.getElementsByTagName('a');
                    for(var i=0; i<links.length; i++){
                        if(links[i].innerText.indexOf('セット') !== -1 || links[i].innerText.indexOf('追加') !== -1){
                            links[i].click();
                            return true;
                        }
                    }
                    return false;
                    """
                    self.driver.execute_script(script_set)
                    entered_count += 1
                    
                    # 次の入力を即座に行うため、waitは最小限に
                    # DOM更新待ちだけ必要
                    time.sleep(0.3) 
                        
                except Exception as e:
                    print(f"Bet processing error: {e}")
                    continue
            
            if entered_count == 0:
                print("No bets entered.")
                return False, "投票データの入力に失敗しました"
                
            # 5. 合計金額入力への遷移
            # JS Click
            try:
                self.driver.execute_script("""
                var links = document.getElementsByTagName('a');
                for(var i=0; i<links.length; i++){
                    if(links[i].innerText.indexOf('入力終了') !== -1){
                        links[i].click();
                        break;
                    }
                }
                """)
                time.sleep(self.WAIT_SEC_LONG)
            except:
                return False, "合計金額入力画面への遷移に失敗しました"
                
            # 6. 合計金額入力
            # ここは重要。合計が合わないとエラーになる
            # 合計を計算
            # 注: 単勝などは1点だが、BOX等は点数計算が必要
            # 簡易的に、渡されたbetsの合計ではなく、画面上の指示に従う必要があるが...
            # 自動入力はリスクがあるので、ユーザーに入力させるか、あるいは計算して入れる
            # ここではフォーカスを当てるだけにする（ユーザー確認のため）
            
            try:
                sum_input = self.driver.find_element(By.NAME, "sum_money") # 仮
                sum_input.click() # フォーカス
                # 可能なら計算値を入れる
                # sum_input.send_keys(str(total_amount))
            except:
                pass
                
            # 7. 確認画面で停止
            if stop_at_confirmation:
                self._save_snapshot("stopped_at_confirmation_pc")
                
                # ウィンドウを最前面に (PC版なのでシンプルに)
                self.driver.switch_to.window(self.driver.current_window_handle)
                try:
                     self.driver.minimize_window()
                     time.sleep(0.5)
                     self.driver.maximize_window()
                except:
                     pass
                     
                return True, "確認画面で停止しました。内容を確認して投票ボタンを押してください。"
            
            # (自動投票する場合のロジックはここに追加)
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
