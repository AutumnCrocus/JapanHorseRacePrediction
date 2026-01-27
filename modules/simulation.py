"""
シミュレーションモジュール
回収率シミュレーションと賭け戦略
"""

import numpy as np
import pandas as pd


class BettingSimulator:
    """賭けシミュレーションクラス"""
    
    def __init__(self, initial_balance: float = 100000):
        """
        初期化
        
        Args:
            initial_balance: 初期資金
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.bet_history = []
    
    def reset(self):
        """シミュレーションをリセット"""
        self.balance = self.initial_balance
        self.bet_history = []
    
    def simulate(self, predictions: pd.DataFrame, return_tables: pd.DataFrame,
                 bet_type: str = 'place', bet_amount: float = 100,
                 threshold: float = 0.5) -> dict:
        """
        賭けシミュレーションを実行
        
        Args:
            predictions: 予測結果（予測確率と実際の着順を含む）
            return_tables: 払戻テーブル
            bet_type: 賭けタイプ ('win', 'place')
            bet_amount: 1回あたりの賭け金
            threshold: 賭けを行う確率の閾値
            
        Returns:
            シミュレーション結果の辞書
        """
        self.reset()
        
        total_bets = 0
        total_wins = 0
        total_returns = 0
        
        # レースごとにシミュレーション
        for race_id in predictions.index.unique():
            race_preds = predictions.loc[race_id] if race_id in predictions.index else None
            
            if race_preds is None or len(race_preds) == 0:
                continue
            
            # 閾値以上の確率の馬に賭ける
            if isinstance(race_preds, pd.Series):
                race_preds = race_preds.to_frame().T
            
            high_prob_horses = race_preds[race_preds['予測確率'] >= threshold]
            
            for _, horse in high_prob_horses.iterrows():
                if self.balance < bet_amount:
                    break
                
                self.balance -= bet_amount
                total_bets += 1
                
                # 的中判定
                actual_rank = horse.get('着順', 99)
                
                if bet_type == 'win' and actual_rank == 1:
                    # 単勝的中
                    odds = horse.get('単勝', 1.0)
                    win_amount = bet_amount * odds
                    self.balance += win_amount
                    total_returns += win_amount
                    total_wins += 1
                
                elif bet_type == 'place' and actual_rank <= 3:
                    # 複勝的中（オッズは単勝の1/3程度と仮定）
                    odds = horse.get('単勝', 1.0) / 3 + 1
                    win_amount = bet_amount * odds
                    self.balance += win_amount
                    total_returns += win_amount
                    total_wins += 1
                
                self.bet_history.append({
                    'race_id': race_id,
                    'horse_number': horse.get('馬番', 0),
                    'predicted_prob': horse.get('予測確率', 0),
                    'actual_rank': actual_rank,
                    'bet_amount': bet_amount,
                    'won': actual_rank <= (1 if bet_type == 'win' else 3)
                })
        
        # 集計
        total_invested = total_bets * bet_amount
        profit = self.balance - self.initial_balance
        recovery_rate = (total_returns / total_invested * 100) if total_invested > 0 else 0
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_bets': total_bets,
            'total_wins': total_wins,
            'win_rate': win_rate,
            'total_invested': total_invested,
            'total_returns': total_returns,
            'profit': profit,
            'recovery_rate': recovery_rate
        }


class BettingPolicy:
    """賭け戦略クラス"""
    
    @staticmethod
    def threshold_policy(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        閾値ベースの賭け戦略
        
        Args:
            predictions: 予測結果
            threshold: 閾値
            
        Returns:
            賭け対象の馬
        """
        return predictions[predictions['予測確率'] >= threshold]
    
    @staticmethod
    def top_n_policy(predictions: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """
        各レースの上位N頭に賭ける戦略
        
        Args:
            predictions: 予測結果
            n: 賭ける頭数
            
        Returns:
            賭け対象の馬
        """
        selected = []
        
        for race_id in predictions.index.unique():
            race_preds = predictions.loc[race_id]
            if isinstance(race_preds, pd.Series):
                race_preds = race_preds.to_frame().T
            
            top_horses = race_preds.nlargest(n, '予測確率')
            selected.append(top_horses)
        
        if selected:
            return pd.concat(selected)
        return pd.DataFrame()
    
    @staticmethod
    def value_betting_policy(predictions: pd.DataFrame, min_value: float = 1.2) -> pd.DataFrame:
        """
        期待値ベースの賭け戦略
        
        Args:
            predictions: 予測結果
            min_value: 最小期待値
            
        Returns:
            賭け対象の馬
        """
        if '単勝' not in predictions.columns:
            return predictions
        
        predictions = predictions.copy()
        predictions['expected_value'] = predictions['予測確率'] * predictions['単勝']
        
        return predictions[predictions['expected_value'] >= min_value]


def run_simulation_report(predictions: pd.DataFrame, return_tables: pd.DataFrame = None) -> dict:
    """
    シミュレーションレポートを生成
    
    Args:
        predictions: 予測結果
        return_tables: 払戻テーブル
        
    Returns:
        レポート辞書
    """
    simulator = BettingSimulator()
    results = {}
    
    # 各戦略でシミュレーション
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        result = simulator.simulate(
            predictions, return_tables,
            bet_type='place',
            threshold=threshold
        )
        results[f'threshold_{threshold}'] = result
    
    return results
