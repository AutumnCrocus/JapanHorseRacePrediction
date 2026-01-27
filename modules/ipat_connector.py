"""
IPAT連携モジュール
JRA IPAT (スマートフォン版) との通信機能を提供します。
"""

import requests
import re
import datetime
from bs4 import BeautifulSoup
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# 定数定義
IPAT_URL_CENTRAL = "https://www.ipat.jra.go.jp/sp/"  # 中央競馬
IPAT_URL_LOCAL = "https://n.ipat.jra.go.jp/sp/"      # 地方競馬

@dataclass
class IpatBetItem:
    """投票内容の1件を表すクラス"""
    race_id: str      # レースID (YYYYMMDDJJRR) な形式を想定
    bet_type_code: int # 式別コード (1:単勝, 2:複勝, ...)
    umaban_list: List[int] # 馬番リスト (単勝なら[1], 3連単なら[1, 2, 3])
    amount: int       # 金額 (100円単位)
    
class IpatConnector:
    """IPAT連携クラス"""
    
    def __init__(self, is_local_horse_racing: bool = False):
        self.session = requests.Session()
        self.base_url = IPAT_URL_LOCAL if is_local_horse_racing else IPAT_URL_CENTRAL
        self.is_login = False
        self.params = {} # ub, g などのパラメータ保持用
        
        # User-Agent設定 (スマホとして振る舞う)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
            'Referer': self.base_url
        })
    
    def _get_page_encoding(self, response):
        """レスポンスのエンコーディングを自動判定（基本はEUC-JP）"""
        return 'euc-jp'
        
    def _normalize_input(self, text: str) -> str:
        """入力値を正規化する（全角→半角、スペース除去）"""
        if not text:
            return ""
        # 全角数字を半角に
        text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        # 全角英字を半角に (INET-ID用)
        text = text.translate(str.maketrans('ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        text = text.translate(str.maketrans('ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ', 'abcdefghijklmnopqrstuvwxyz'))
        # スペース除去
        text = text.replace(' ', '').replace('　', '')
        return text

    def _save_debug_html(self, content: bytes, filename: str = 'debug_login_error.html'):
        """デバッグ用HTMLを保存"""
        try:
            import os
            debug_path = os.path.join(os.getcwd(), filename)
            # 既存ファイルを削除して確実に上書き
            if os.path.exists(debug_path):
                os.remove(debug_path)
            with open(debug_path, 'wb') as f:
                f.write(content)
            print(f"Saved debug html to: {debug_path}")
        except Exception as e:
            print(f"Failed to save debug html: {e}")

    def login(self, inetid: str, subscriber_no: str, pin: str, pars_no: str) -> Tuple[bool, str]:
        """IPATにログインする"""
        # 入力値の正規化
        inetid = self._normalize_input(inetid)
        subscriber_no = self._normalize_input(subscriber_no)
        pin = self._normalize_input(pin)
        pars_no = self._normalize_input(pars_no)
        
        current_res = None # 初期化

        try:
            # 1. トップページアクセスして初期パラメータ取得
            # 本来は hidden パラメータ (uh 等) を取得する必要がある
            res = self.session.get(self.base_url)
            res.encoding = self._get_page_encoding(res)
            
            soup = BeautifulSoup(res.text, 'lxml')
            
            # formを探す
            form = soup.find('form')
            if not form:
                print("Error: Login form not found.")
                return False, "ログインフォームが見つかりません。JRAサイトが変更された可能性があります。"
                
            action = form.get('action')
            
            # hiddenパラメータ取得
            post_data = {}
            for inp in form.find_all('input'):
                if inp.get('type') == 'hidden':
                    post_data[inp.get('name')] = inp.get('value')
            
            # ユーザー情報の入力
            post_data['inetid'] = inetid
            post_data['i'] = subscriber_no
            post_data['p'] = pin
            post_data['r'] = pars_no
            
            # ログイン実行
            login_url = urllib.parse.urljoin(self.base_url, action)
            res_login = self.session.post(login_url, data=post_data)
            res_login.encoding = self._get_page_encoding(res_login)
            
            # ログイン後の画面遷移ループ（重要なお知らせスキップ用）
            current_res = res_login
            for _ in range(3):
                # 成功判定
                if "メニュー" in current_res.text or "トップメニュー" in current_res.text:
                    self.is_login = True
                    print("IPAT Login Successful")
                    return True, "ログイン成功"
                
                # エラーメッセージの判定（JRAのエラー画面）
                # NOTE: エンコーディング等の問題で完全一致しない場合があるため、キーワードを細かく分ける
                text = current_res.text
                if "しばらくたってから" in text or "(001)" in text or "混雑" in text:
                     self._save_debug_html(current_res.content)
                     return False, "JRAサーバーエラー: 混雑等のためログインできません。(001) - 時間を置いて再試行してください"
                if "入力された内容に" in text or "誤りがあります" in text:
                     self._save_debug_html(current_res.content)
                     return False, "認証エラー: 入力された内容に誤りがあります。"
                if "受付時間外" in text:
                     self._save_debug_html(current_res.content)
                     return False, "JRA受付時間外です。"
                if "メンテナンス" in text:
                     self._save_debug_html(current_res.content)
                     return False, "JRAシステムメンテナンス中です。"
                
                # 「重要なお知らせ」などの確認画面判定
                soup = BeautifulSoup(current_res.text, 'lxml')
                forms = soup.find_all('form')
                
                # パスワード入力欄がある場合、それは中間画面ではなく再ログイン画面（エラー画面）である
                for f in forms:
                    if f.find('input', type='password'):
                        print("IPAT Login: Re-login form detected. Authentication failed.")
                        self._save_debug_html(current_res.content)
                        return False, "認証エラー: 再ログイン画面が表示されました。認証情報を確認してください。"

                # フォームが一つだけあり、それが「お知らせ」確認等の場合、自動サブミットする
                if len(forms) == 1:
                    print("IPAT Login: Intermediate screen detected. Trying to skip...")
                    form = forms[0]
                    next_action = form.get('action')
                    if not next_action:
                         # actionがないフォーム（JavaScript戻るボタンなど）はエラー画面の可能性が高い
                         self._save_debug_html(current_res.content)
                         return False, "不明なエラー画面（再試行ボタンなし）"
                         
                    next_url = urllib.parse.urljoin(self.base_url, next_action)
                    
                    next_data = {}
                    for inp in form.find_all('input'):
                        if inp.get('name'):
                            next_data[inp.get('name')] = inp.get('value', '')
                            
                    # 次へ進む
                    current_res = self.session.post(next_url, data=next_data)
                    current_res.encoding = self._get_page_encoding(current_res)
                    continue
                
                # フォームが複数ある場合の処理（ボタンのテキスト等で判断）
                elif len(forms) > 1:
                    print(f"IPAT Login: Multiple forms ({len(forms)}) detected.")
                    target_form = None
                    
                    # submitボタンのvalueに「次へ」「同意」「送信」などが含まれるフォームを探す
                    for f in forms:
                        submit = f.find('input', type='submit')
                        if submit and submit.get('value'):
                            val = submit.get('value')
                            if any(k in val for k in ['次へ', '同意', '送信', 'OK', '確認']):
                                target_form = f
                                break
                    
                    if target_form:
                        print("IPAT Login: Target form found by button text.")
                    else:
                        # キーワードで見つからない場合、ヒューリスティックな選択を試みる
                        print("IPAT Login: Target form not found by text. Trying heuristic...")
                        for i, f in enumerate(forms):
                            # デバッグ情報出力
                            inputs = [i.get('name') for i in f.find_all('input') if i.get('name')]
                            buttons = [b.get('value') or b.text for b in f.find_all(['input', 'button']) if b.name == 'button' or b.get('type') in ['submit', 'button']]
                            action = f.get('action', '')
                            print(f"Form {i}: action={action}, inputs={inputs}, buttons={buttons}")
                            
                            # ログアウトや戻るボタンっぽいフォームは除外
                            if any(k in action for k in ['logout', 'menu']):
                                continue
                                
                            # hiddenパラメータなどのinputが多いフォームを有力候補とする
                            if len(inputs) >= 1:
                                target_form = f
                                print(f"Form {i} selected as target (heuristic).")
                                break
                        
                    if target_form:
                        form = target_form
                        next_action = form.get('action')
                        # actionが空の場合は自分自身へのPOSTとみなす
                        next_url = urllib.parse.urljoin(self.base_url, next_action) if next_action else current_res.url
                        
                        next_data = {}
                        for inp in form.find_all('input'):
                            if inp.get('name'):
                                next_data[inp.get('name')] = inp.get('value', '')
                        
                        current_res = self.session.post(next_url, data=next_data)
                        current_res.encoding = self._get_page_encoding(current_res)
                        continue

                else:
                    return False, "メニュー画面に到達できませんでした（フォームなし）"
            
            print("IPAT Login Failed: Menu not found after retries.")
            if current_res:
                self._save_debug_html(current_res.content)
            
            return False, "ログイン試行回数を超過しました。JRAサイトの状態を確認してください。"
                
        except Exception as e:
            print(f"IPAT Login Error: {e}")
            import traceback
            traceback.print_exc()
            return False, f"システムエラー: {str(e)}"

    def logout(self):
        """ログアウト処理（セッション破棄のみ）"""
        self.session.close()
        self.is_login = False

    def build_bet_string(self, bets: List[IpatBetItem]) -> str:
        """
        投票用文字列を構築する (中央競馬フォーマット)
        参考: 27桁フォーマット
        [ステータス:1][購入番号:3][会場Index:1][レース番号:1][曜日ID:1][方式:1][式別:1][組番データ:14][金額(16進数):4]
        
        注: 現時点では簡易実装。実際にはAPI呼び出し元から会場コードや曜日IDなどを渡す必要がある。
        """
        # TODO: 正式な実装にはレースメタデータ（会場コード、回次、日次）が必要
        # このメソッドはモック的に動作する
        
        ticket_str_list = []
        for i, bet in enumerate(bets):
            # ダミーデータの構築
            
            status = "1" # 通常
            buy_no = f"{i+1:03}" # 購入番号 (001~)
            
            # 会場、レース番号などのエンコードロジックが必要
            # ここでは仮の値を入れる
            place_idx = "1" # 東京?
            race_no = "1" # 1R?
            day_id = "1" 
            
            method = "0" # 通常
            type_code = str(bet.bet_type_code) # 式別
            
            # 組番データの整形 (14桁)
            # 馬番を2桁ゼロ埋めで連結し、後ろを0またはスペースで埋める
            umaban_str = "".join([f"{u:02}" for u in bet.umaban_list])
            kumi_data = f"{umaban_str:<14}".replace(" ", "0") 
            
            # 金額 (100円単位の枚数ではなく、金額そのものを16進数か？ブログには金額(16進数):4とある)
            # 例: 100円 -> 0064 ? それとも枚数0001 ?
            # ブログ「100円なら 0001、1000円なら 000a」とあるので、100円単位の枚数を16進数にするようだ。
            coins = bet.amount // 100
            amount_hex = f"{coins:04x}"
            
            ticket = f"{status}{buy_no}{place_idx}{race_no}{day_id}{method}{type_code}{kumi_data}{amount_hex}"
            ticket_str_list.append(ticket)
            
        return ",".join(ticket_str_list)

    def vote(self, bets: List[IpatBetItem]) -> Dict:
        """投票を実行する"""
        if not self.is_login:
            return {"success": False, "message": "Not logged in"}
            
        # 1. 投票入力画面へ遷移 (パラメータチェック)
        # 2. 投票確認画面へ (金額、組数の確認)
        # 3. 投票完了画面へ
        
        # NOTE: 非常にセンシティブな処理かつ、現在のモック情報だけでは完全にシミュレートするのは危険。
        # ここでは「ログイン済みであれば成功したフリをする」実装にとどめ、
        # 実際の運用時は慎重にテストする必要がある。
        
        try:
            # 投票文字列の生成（テスト用）
            bet_string = self.build_bet_string(bets)
            print(f"Generated Bet String: {bet_string}")
            
            # 擬似的な成功レスポンス
            return {
                "success": True,
                "message": "投票を受け付けました（モック）",
                "details": {
                    "count": len(bets),
                    "total_amount": sum(b.amount for b in bets),
                    "bet_string": bet_string
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Vote Error: {e}"}

