import sys
import os
import pandas as pd
import pytest

# プロジェクトルートをパスに追加してモジュールをインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.betting_allocator import BettingAllocator

class TestBettingAllocator:
    def setup_method(self):
        # テスト用のダミーデータ
        data = [
            {'horse_number': 1, 'probability': 0.60},
            {'horse_number': 2, 'probability': 0.30},
            {'horse_number': 3, 'probability': 0.15},
            {'horse_number': 4, 'probability': 0.10},
            {'horse_number': 5, 'probability': 0.08},
            {'horse_number': 6, 'probability': 0.05},
            {'horse_number': 7, 'probability': 0.02},
            {'horse_number': 8, 'probability': 0.01},
        ]
        self.df_preds = pd.DataFrame(data)

    def test_allocate_budget_basic(self):
        """基本機能テスト: 予算内で買い目が生成されるか"""
        budget = 1000
        recs = BettingAllocator.allocate_budget(self.df_preds, budget)
        
        total = sum(r['total_amount'] for r in recs)
        assert total <= budget, f"予算超過: {total} > {budget}"
        assert len(recs) > 0, "買い目が生成されていません"

    def test_allocate_various_budgets(self):
        """複数バリエーションの予算テスト"""
        budgets = [1000, 3000, 5000, 10000]
        
        for b in budgets:
            recs = BettingAllocator.allocate_budget(self.df_preds, b)
            if not recs:
                continue
            total = sum(r['total_amount'] for r in recs)
            assert total <= b, f"予算 {b} 円の場合に超過発生: {total}"
            
            # 必須フィールドの確認
            for r in recs:
                assert 'bet_type' in r
                assert 'method' in r
                assert 'total_amount' in r

    def test_zero_budget(self):
        """予算0円のケース"""
        recs = BettingAllocator.allocate_budget(self.df_preds, 0)
        assert recs == [], "予算0円なら推奨なしであるべき"

if __name__ == "__main__":
    # 直接実行時はpytestを起動する形にするか、あるいは単純実行
    # ここでは便宜上、クラスをインスタンス化して実行する簡易コード
    t = TestBettingAllocator()
    t.setup_method()
    t.test_allocate_budget_basic()
    t.test_allocate_various_budgets()
    t.test_zero_budget()
    print("All manual checks passed.")
