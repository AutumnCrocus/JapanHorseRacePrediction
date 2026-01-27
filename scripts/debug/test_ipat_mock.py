import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import unittest
from unittest.mock import MagicMock, patch
from modules.ipat_connector import IpatConnector, IpatBetItem

class TestIpatConnector(unittest.TestCase):
    
    def setUp(self):
        self.connector = IpatConnector()
        
    @patch('modules.ipat_connector.requests.Session')
    def test_login_success(self, mock_session_cls):
        """ログイン成功時のテスト"""
        # セッションのモック設定
        mock_session = mock_session_cls.return_value
        self.connector.session = mock_session
        
        # 1. トップページ取得のモック
        mock_res_top = MagicMock()
        mock_res_top.text = """
        <html><body>
        <form action="/sp/login.cgi" method="post">
            <input type="hidden" name="uh" value="dummy_uh">
            <input type="hidden" name="g" value="730">
        </form>
        </body></html>
        """
        mock_session.get.return_value = mock_res_top
        
        # 2. ログインPOSTのモック
        mock_res_login = MagicMock()
        mock_res_login.text = "<html><body>トップメニュー</body></html>"
        mock_session.post.return_value = mock_res_login
        
        # 実行
        result = self.connector.login("INETID", "1234567890", "1234", "5678")
        
        # 検証
        self.assertTrue(result)
        self.assertTrue(self.connector.is_login)
        
        # 正しいパラメータでPOSTされたか確認
        post_args = mock_session.post.call_args
        url = post_args[0][0]
        data = post_args[1]['data']
        
        self.assertIn('/sp/login.cgi', url)
        self.assertEqual(data['uh'], 'dummy_uh')
        self.assertEqual(data['inetid'], 'INETID')
        self.assertEqual(data['i'], '1234567890')
        self.assertEqual(data['p'], '1234')
        self.assertEqual(data['r'], '5678')
        
    def test_build_bet_string(self):
        """馬券文字列生成のテスト"""
        bets = [
            IpatBetItem(
                race_id="202601240101",
                bet_type_code=1, # 単勝
                umaban_list=[5],
                amount=1000
            ),
            IpatBetItem(
                race_id="202601240101",
                bet_type_code=4, # 馬連
                umaban_list=[1, 2],
                amount=500
            )
        ]
        
        bet_string = self.connector.build_bet_string(bets)
        print(f"Test Bet String: {bet_string}")
        
        tickets = bet_string.split(',')
        self.assertEqual(len(tickets), 2)
        
        # 1枚目: 単勝 1000円 (000a枚)
        # [1][001][1][1][1][0][1][0500...][000a] (27桁)
        t1 = tickets[0]
        self.assertEqual(len(t1), 27)
        self.assertEqual(t1[-4:], '000a') # 10枚
        self.assertEqual(t1[8], '1') # 式別: 単勝 (Index 8)
        self.assertIn('05', t1) # 馬番5
        
        # 2枚目: 馬連 500円 (0005枚)
        # [1][002][1][1][1][0][4][0102...][0005] (27桁)
        t2 = tickets[1]
        self.assertEqual(len(t2), 27)
        self.assertEqual(t2[-4:], '0005') # 5枚
        self.assertEqual(t2[8], '4') # 式別: 馬連 (Index 8)
        self.assertIn('0102', t2) # 馬番1-2

if __name__ == '__main__':
    unittest.main()
