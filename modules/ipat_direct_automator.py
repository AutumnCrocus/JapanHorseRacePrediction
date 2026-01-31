"""
IPAT直接連携モジュール（Selenium版）
JRA IPATに直接アクセスして投票画面を自動操作するモジュール
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
from typing import List, Dict, Any, Optional
import os
import datetime

class IpatDirectAutomator:
    """IPAT直接連携クラス（Selenium版）"""
    
    def __init__(self):
        """初期化"""
        self.driver = None
        self.wait_timeout = 10  # デフォルトの待機時間（秒）
        
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
    
    def login(self, inetid: str, subscriber_no: str, pin: str, pars_no: str) -> tuple[bool, str]:
        """
        IPATログイン画面で認証を実行
        
        Args:
            inetid: INET-ID
            subscriber_no: 加入者番号
            pin: 暗証番号
            pars_no: P-ARS番号
            
        Returns:
            (成功フラグ, メッセージ)
        """
        try:
            # Chromeオプション設定
            options = webdriver.ChromeOptions()
            options.add_argument('--start-maximized')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            # ドライバ起動
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            wait = WebDriverWait(self.driver, self.wait_timeout)
            
            print("IPATログインページにアクセス中...")
            self.driver.get("https://www.ipat.jra.go.jp/sp/")
            time.sleep(2)
            
            # ログインフォームの入力
            print("認証情報を入力中...")
            
            # INET-ID入力（フィールドが存在する場合）
            try:
                inetid_field = wait.until(EC.presence_of_element_located((By.NAME, "inetid")))
                inetid_field.clear()
                inetid_field.send_keys(inetid)
            except:
                print("Warning: INET-ID field not found (may be optional)")
            
            # 加入者番号
            subscriber_field = wait.until(EC.presence_of_element_located((By.NAME, "i")))
            subscriber_field.clear()
            subscriber_field.send_keys(subscriber_no)
            
            # 暗証番号
            pin_field = self.driver.find_element(By.NAME, "p")
            pin_field.clear()
            pin_field.send_keys(pin)
            
            # P-ARS番号
            pars_field = self.driver.find_element(By.NAME, "r")
            pars_field.clear()
            pars_field.send_keys(pars_no)
            
            self._save_debug_screenshot(self.driver, "before_login")
            
            # ログインボタンをクリック
            print("ログインボタンをクリック中...")
            login_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            login_button.click()
            
            time.sleep(3)
            self._save_debug_screenshot(self.driver, "after_login")
            
            # ログイン成功確認（メニュー画面に到達したか）
            try:
                # メニュー画面の特徴的な要素を探す
                wait.until(lambda d: "メニュー" in d.page_source or "トップメニュー" in d.page_source)
                print("IPATログイン成功")
                return True, "ログイン成功"
            except:
                # エラーメッセージを確認
                page_text = self.driver.page_source
                if "誤り" in page_text or "エラー" in page_text:
                    self._save_debug_screenshot(self.driver, "login_error")
                    return False, "認証エラー: 入力された内容に誤りがあります"
                elif "混雑" in page_text:
                    return False, "JRAサーバーエラー: 混雑のため接続できません"
                else:
                    return False, "ログイン失敗: メニュー画面に到達できませんでした"
                    
        except Exception as e:
            print(f"IPATログインエラー: {e}")
            if self.driver:
                self._save_debug_screenshot(self.driver, "login_exception")
            return False, f"システムエラー: {str(e)}"
    
    def navigate_to_race_bet_page(self, race_id: str) -> tuple[bool, str]:
        """
        レースIDから投票画面へ遷移
        
        Args:
            race_id: レースID（例: 202601310101）
            
        Returns:
            (成功フラグ, メッセージ)
        """
        if not self.driver:
            return False, "ドライバが初期化されていません"
            
        try:
            print(f"レース {race_id} の投票画面へ遷移中...")
            
            # レースIDから会場、レース番号を解析
            # race_id形式: YYYYMMDDKKRR (YYYY=年, MM=月, DD=日, KK=会場+回次+日次, RR=レース番号)
            year = race_id[0:4]
            month = race_id[4:6]
            day = race_id[6:8]
            race_info = race_id[8:10]  # 会場コード等
            race_num = race_id[10:12]
            
            # IPAT投票画面へのURL構築（推測）
            # 注: 実際のIPATのURL構造は非公開のため、ログイン後の画面から手動で確認する必要がある
            bet_url = f"https://www.ipat.jra.go.jp/sp/bet/?date={year}{month}{day}&race_info={race_info}&race_num={race_num}"
            
            print(f"投票画面URL: {bet_url}")
            self.driver.get(bet_url)
            time.sleep(2)
            
            self._save_debug_screenshot(self.driver, "bet_page")
            
            # 投票画面が表示されたか確認
            wait = WebDriverWait(self.driver, self.wait_timeout)
            try:
                # 投票画面の特徴的な要素（券種タブなど）を探す
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='type'], button[class*='bet'], input[name*='umaban']")))
                print("投票画面に到達しました")
                return True, "投票画面表示成功"
            except:
                # エラーの場合、手動でレース選択画面へ誘導するメッセージを返す
                return False, "投票画面への自動遷移に失敗しました。手動でレースを選択してください。"
                
        except Exception as e:
            print(f"投票画面遷移エラー: {e}")
            if self.driver:
                self._save_debug_screenshot(self.driver, "navigate_error")
            return False, f"システムエラー: {str(e)}"
    
    def fill_bet_form(self, bets: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        買い目を自動入力（券種、馬番、金額）
        
        Args:
            bets: 買い目リスト
                [
                    {
                        'type': '単勝',
                        'horses': [1],
                        'amount': 100
                    },
                    {
                        'type': '馬連',
                        'horses': [1, 2],
                        'amount': 100,
                        'method': 'box'  # 'normal', 'box', 'formation'
                    }
                ]
                
        Returns:
            (成功フラグ, メッセージ)
        """
        if not self.driver:
            return False, "ドライバが初期化されていません"
            
        try:
            success_count = 0
            
            for i, bet in enumerate(bets):
                print(f"買い目 {i+1}/{len(bets)} を入力中: {bet}")
                
                # 券種を選択
                bet_type = bet.get('type', '単勝')
                if not self._select_bet_type(bet_type):
                    print(f"Warning: 券種 '{bet_type}' の選択に失敗しました")
                    continue
                
                # 投票方式を選択（ボックス、フォーメーション等）
                method = bet.get('method', 'normal')
                if method != 'normal':
                    self._select_betting_method(method)
                
                # 馬番を入力
                horses = bet.get('horses', [])
                if not self._enter_horse_numbers(horses, method):
                    print(f"Warning: 馬番の入力に失敗しました")
                    continue
                
                # 金額を入力
                amount = bet.get('amount', 100)
                if not self._enter_amount(amount):
                    print(f"Warning: 金額の入力に失敗しました")
                    continue
                
                # 追加ボタンをクリック（投票リストに追加）
                if self._click_add_button():
                    success_count += 1
                    print(f"買い目 {i+1} を投票リストに追加しました")
                else:
                    print(f"Warning: 買い目 {i+1} の追加に失敗しました")
                
                time.sleep(1)  # 次の買い目入力前に少し待機
            
            self._save_debug_screenshot(self.driver, "after_fill_all")
            
            if success_count > 0:
                return True, f"{success_count}/{len(bets)} 件の買い目を入力しました。投票確定は手動で行ってください。"
            else:
                return False, "買い目の入力に失敗しました"
                
        except Exception as e:
            print(f"買い目入力エラー: {e}")
            if self.driver:
                self._save_debug_screenshot(self.driver, "fill_error")
            return False, f"システムエラー: {str(e)}"
    
    def _select_bet_type(self, bet_type: str) -> bool:
        """券種タブを選択"""
        try:
            # 券種マッピング
            type_map = {
                '単勝': 'tan',
                '複勝': 'fuku',
                '枠連': 'waku',
                '馬連': 'umaren',
                'ワイド': 'wide',
                '馬単': 'umatan',
                '3連複': 'sanrenpuku',
                '3連単': 'sanrentan'
            }
            
            type_code = type_map.get(bet_type, 'tan')
            
            # セレクタのパターンを試す
            selectors = [
                f"a[href*='{type_code}']",
                f"button[data-type='{type_code}']",
                f"input[value='{type_code}']",
                f"//a[contains(text(), '{bet_type}')]",
                f"//button[contains(text(), '{bet_type}')]"
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith('//'):
                        # XPath
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        # CSS Selector
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    element.click()
                    time.sleep(0.5)
                    print(f"券種 '{bet_type}' を選択しました")
                    return True
                except:
                    continue
            
            print(f"Warning: 券種 '{bet_type}' のタブが見つかりませんでした")
            return False
            
        except Exception as e:
            print(f"券種選択エラー: {e}")
            return False
    
    def _select_betting_method(self, method: str) -> bool:
        """投票方式を選択（ボックス、フォーメーション等）"""
        try:
            method_map = {
                'box': 'ボックス',
                'formation': 'フォーメーション',
                'nagashi': 'ながし'
            }
            
            method_text = method_map.get(method, method)
            
            # ラジオボタンまたはタブを探す
            selectors = [
                f"//input[@type='radio' and contains(@value, '{method}')]",
                f"//label[contains(text(), '{method_text}')]",
                f"//a[contains(text(), '{method_text}')]"
            ]
            
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    element.click()
                    time.sleep(0.5)
                    print(f"投票方式 '{method_text}' を選択しました")
                    return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"投票方式選択エラー: {e}")
            return False
    
    def _enter_horse_numbers(self, horses: List[int], method: str = 'normal') -> bool:
        """馬番を入力"""
        try:
            # チェックボックス方式
            for horse_num in horses:
                selectors = [
                    f"input[type='checkbox'][value='{horse_num}']",
                    f"input[type='checkbox'][name*='umaban'][value='{horse_num}']",
                    f"//label[contains(text(), '{horse_num}')]//input[@type='checkbox']",
                    f"//td[text()='{horse_num}']//input[@type='checkbox']"
                ]
                
                found = False
                for selector in selectors:
                    try:
                        if selector.startswith('//'):
                            element = self.driver.find_element(By.XPATH, selector)
                        else:
                            element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        
                        if not element.is_selected():
                            element.click()
                        found = True
                        print(f"馬番 {horse_num} を選択しました")
                        break
                    except:
                        continue
                
                if not found:
                    print(f"Warning: 馬番 {horse_num} のチェックボックスが見つかりませんでした")
            
            return True
            
        except Exception as e:
            print(f"馬番入力エラー: {e}")
            return False
    
    def _enter_amount(self, amount: int) -> bool:
        """金額を入力"""
        try:
            # 金額入力フィールドを探す
            selectors = [
                "input[name*='amount']",
                "input[name*='kingaku']",
                "input[type='number']",
                "input[type='text'][placeholder*='金額']"
            ]
            
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    element.clear()
                    element.send_keys(str(amount))
                    print(f"金額 {amount}円 を入力しました")
                    return True
                except:
                    continue
            
            print("Warning: 金額入力フィールドが見つかりませんでした")
            return False
            
        except Exception as e:
            print(f"金額入力エラー: {e}")
            return False
    
    def _click_add_button(self) -> bool:
        """追加ボタンをクリック"""
        try:
            # 追加ボタンを探す
            selectors = [
                "input[type='submit'][value*='追加']",
                "button[type='submit']",
                "input[type='button'][value*='追加']",
                "//button[contains(text(), '追加')]",
                "//input[@type='submit' and contains(@value, '追加')]"
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    element.click()
                    time.sleep(1)
                    print("追加ボタンをクリックしました")
                    return True
                except:
                    continue
            
            print("Warning: 追加ボタンが見つかりませんでした")
            return False
            
        except Exception as e:
            print(f"追加ボタンクリックエラー: {e}")
            return False
    
    def close(self):
        """ブラウザを閉じる（手動操作完了後）"""
        if self.driver:
            print("注意: ブラウザは手動で閉じてください（投票確定後）")
            # self.driver.quit()  # 自動では閉じない
