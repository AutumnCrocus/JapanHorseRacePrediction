"""
Kelly Criterion Strategy Module
ケリー基準に基づく最適資金配分戦略

Kelly Criterion: f* = (p*b - q) / b
- p = 勝率 (予測確率)
- q = 敗率 (1 - p)
- b = オッズ - 1 (純利益倍率)
- f* = 資金の最適賭け比率
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class KellyStrategy:
    """
    ケリー基準に基づく資金配分戦略クラス
    
    特徴:
    - 期待値が正のベットのみ推奨
    - エッジが大きいほど多く、小さいほど少なく賭ける
    - Half-Kellyオプションでリスク軽減
    """
    
    def __init__(self, 
                 use_half_kelly: bool = True,
                 min_edge: float = 0.05,
                 max_fraction: float = 0.25,
                 min_bet: int = 100):
        """
        Args:
            use_half_kelly: Half-Kelly（計算値の50%）を使用するか
            min_edge: 最小エッジ閾値（これ以下は賭けない）
            max_fraction: 1ベットあたりの最大資金比率
            min_bet: 最小賭け金（100円単位）
        """
        self.use_half_kelly = use_half_kelly
        self.min_edge = min_edge
        self.max_fraction = max_fraction
        self.min_bet = min_bet
    
    def calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """
        ケリー基準で最適賭け比率を計算
        
        Args:
            win_prob: 勝率（0-1）
            odds: オッズ（1.5なら1.5倍）
            
        Returns:
            最適資金比率（0-1）、負のエッジは0を返す
        """
        if odds <= 1.0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
            
        b = odds - 1  # 純利益倍率
        q = 1 - win_prob  # 敗率
        
        # Kelly Criterion: f* = (p*b - q) / b
        f = (win_prob * b - q) / b
        
        # 負のエッジは賭けない
        if f <= 0:
            return 0.0
        
        # Half-Kelly適用
        if self.use_half_kelly:
            f *= 0.5
        
        # 最大比率でキャップ
        f = min(f, self.max_fraction)
        
        return f
    
    def calculate_edge(self, win_prob: float, odds: float) -> float:
        """
        期待値（エッジ）を計算
        
        Args:
            win_prob: 勝率
            odds: オッズ
            
        Returns:
            期待値（1.0 = 100%、1.2 = 120%回収期待）
        """
        return win_prob * odds
    
    def allocate_budget(self, 
                       predictions: pd.DataFrame,
                       total_budget: int,
                       bet_type: str = 'tan') -> List[Dict[str, Any]]:
        """
        予測データから最適資金配分を計算
        
        Args:
            predictions: 予測データフレーム（prob, odds列必須）
            total_budget: 総予算
            bet_type: 'tan'(単勝) or 'fuku'(複勝)
            
        Returns:
            推奨ベットリスト
        """
        bets = []
        
        prob_col = 'prob' if 'prob' in predictions.columns else 'win_prob'
        odds_col = 'tan_odds' if bet_type == 'tan' else 'fuku_odds_mid'
        
        if odds_col not in predictions.columns:
            odds_col = 'odds'
        
        for idx, row in predictions.iterrows():
            prob = row.get(prob_col, 0)
            odds = row.get(odds_col, 0)
            
            if prob <= 0 or odds <= 1:
                continue
            
            # エッジ計算
            edge = self.calculate_edge(prob, odds)
            
            # 最小エッジ未満はスキップ
            if edge < 1 + self.min_edge:
                continue
            
            # Kelly比率計算
            fraction = self.calculate_kelly_fraction(prob, odds)
            
            if fraction <= 0:
                continue
            
            # 賭け金計算（100円単位に丸め）
            bet_amount = int(total_budget * fraction / 100) * 100
            
            if bet_amount < self.min_bet:
                continue
            
            horse_num = row.get('馬番', row.get('horse_number', idx + 1))
            
            bets.append({
                'type': '単勝' if bet_type == 'tan' else '複勝',
                'method': 'SINGLE',
                'horse_numbers': [int(horse_num)],
                'total_amount': bet_amount,
                'odds': odds,
                'prob': prob,
                'edge': edge,
                'kelly_fraction': fraction,
                'reason': f"Kelly推奨: 期待値{edge:.2f}倍, 資金比率{fraction*100:.1f}%"
            })
        
        # 予算オーバーの場合は比率で調整
        total_allocated = sum(b['total_amount'] for b in bets)
        if total_allocated > total_budget and bets:
            scale = total_budget / total_allocated
            for bet in bets:
                bet['total_amount'] = max(self.min_bet, 
                                          int(bet['total_amount'] * scale / 100) * 100)
        
        # エッジ順でソート
        bets.sort(key=lambda x: x['edge'], reverse=True)
        
        return bets
    
    def simulate_race(self,
                      predictions: pd.DataFrame,
                      result_rank: Dict[int, int],
                      total_budget: int,
                      bet_type: str = 'tan') -> Dict[str, Any]:
        """
        1レースをシミュレート
        
        Args:
            predictions: 予測データ
            result_rank: {馬番: 着順} の辞書
            total_budget: 予算
            bet_type: 馬券タイプ
            
        Returns:
            シミュレーション結果
        """
        bets = self.allocate_budget(predictions, total_budget, bet_type)
        
        total_bet = sum(b['total_amount'] for b in bets)
        total_return = 0
        
        for bet in bets:
            horse_num = bet['horse_numbers'][0]
            rank = result_rank.get(horse_num, 99)
            
            if bet_type == 'tan':
                # 単勝: 1着のみ
                if rank == 1:
                    total_return += bet['total_amount'] * bet['odds']
            else:
                # 複勝: 3着以内
                if rank <= 3:
                    # 複勝オッズは変動するため、中央値で計算
                    total_return += bet['total_amount'] * bet['odds']
        
        return {
            'total_bet': total_bet,
            'total_return': total_return,
            'profit': total_return - total_bet,
            'recovery_rate': total_return / total_bet if total_bet > 0 else 0,
            'num_bets': len(bets),
            'bets': bets
        }


class KellyBacktester:
    """
    ケリー戦略のバックテスト実行クラス
    """
    
    def __init__(self, strategy: KellyStrategy = None):
        self.strategy = strategy or KellyStrategy()
        self.results = []
    
    def run_backtest(self,
                    race_data: List[Dict],
                    budget_patterns: List[int] = [5000, 10000, 20000],
                    bet_type: str = 'tan') -> pd.DataFrame:
        """
        バックテスト実行
        
        Args:
            race_data: レースデータのリスト
            budget_patterns: テストする予算パターン
            bet_type: 馬券タイプ
            
        Returns:
            結果サマリーDataFrame
        """
        results = []
        
        for budget in budget_patterns:
            total_bet = 0
            total_return = 0
            wins = 0
            races = 0
            
            for race in race_data:
                predictions = race.get('predictions', pd.DataFrame())
                result_rank = race.get('result_rank', {})
                
                if predictions.empty:
                    continue
                
                sim = self.strategy.simulate_race(
                    predictions, result_rank, budget, bet_type
                )
                
                total_bet += sim['total_bet']
                total_return += sim['total_return']
                if sim['profit'] > 0:
                    wins += 1
                races += 1
            
            results.append({
                'budget': budget,
                'races': races,
                'total_bet': total_bet,
                'total_return': total_return,
                'profit': total_return - total_bet,
                'recovery_rate': total_return / total_bet if total_bet > 0 else 0,
                'win_rate': wins / races if races > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def compare_strategies(self,
                          race_data: List[Dict],
                          strategies: Dict[str, KellyStrategy],
                          budget: int = 10000) -> pd.DataFrame:
        """
        複数戦略の比較
        
        Args:
            race_data: レースデータ
            strategies: {名前: KellyStrategy} の辞書
            budget: 予算
            
        Returns:
            比較結果DataFrame
        """
        results = []
        
        for name, strategy in strategies.items():
            total_bet = 0
            total_return = 0
            
            for race in race_data:
                predictions = race.get('predictions', pd.DataFrame())
                result_rank = race.get('result_rank', {})
                
                if predictions.empty:
                    continue
                
                sim = strategy.simulate_race(predictions, result_rank, budget)
                total_bet += sim['total_bet']
                total_return += sim['total_return']
            
            results.append({
                'strategy': name,
                'total_bet': total_bet,
                'total_return': total_return,
                'recovery_rate': total_return / total_bet if total_bet > 0 else 0
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # テスト用のサンプルデータ
    print("=== Kelly Criterion Strategy Test ===")
    
    strategy = KellyStrategy(use_half_kelly=True)
    
    # テストケース: 勝率30%, オッズ5.0
    prob, odds = 0.30, 5.0
    fraction = strategy.calculate_kelly_fraction(prob, odds)
    edge = strategy.calculate_edge(prob, odds)
    
    print(f"勝率: {prob*100:.0f}%, オッズ: {odds:.1f}倍")
    print(f"期待値: {edge:.2f}倍")
    print(f"Kelly比率: {fraction*100:.2f}%")
    print(f"1万円中の賭け金: {int(10000 * fraction / 100) * 100}円")
    
    # 追加テストケース
    test_cases = [
        (0.20, 8.0),
        (0.10, 15.0),
        (0.50, 2.5),
        (0.15, 10.0),
    ]
    
    print("\n=== 各ケースのKelly計算 ===")
    for p, o in test_cases:
        f = strategy.calculate_kelly_fraction(p, o)
        e = strategy.calculate_edge(p, o)
        print(f"勝率{p*100:.0f}% オッズ{o:.1f}x -> 期待値{e:.2f}x Kelly{f*100:.1f}%")
