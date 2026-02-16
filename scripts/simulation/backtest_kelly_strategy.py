"""
Kelly Strategy Backtest Script
2024-2025年データでKelly戦略 vs 既存戦略を比較

予算パターン: 5,000円 / 10,000円 / 20,000円
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from modules.kelly_strategy import KellyStrategy


def load_historical_data(data_dir: str, years: list = [2024, 2025]) -> pd.DataFrame:
    """
    過去データを読み込み
    
    Args:
        data_dir: データディレクトリ
        years: 対象年リスト
        
    Returns:
        結合されたDataFrame
    """
    all_data = []
    
    for year in years:
        # レース結果データ
        result_path = os.path.join(data_dir, f"race_results_{year}.csv")
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            all_data.append(df)
            print(f"Loaded: {result_path} ({len(df)} records)")
    
    if not all_data:
        # CSVがなければPKLを試す
        for year in years:
            pkl_path = os.path.join(data_dir, f"race_results_{year}.pkl")
            if os.path.exists(pkl_path):
                df = pd.read_pickle(pkl_path)
                all_data.append(df)
                print(f"Loaded: {pkl_path} ({len(df)} records)")
    
    if not all_data:
        print("No historical data found.")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def simulate_kelly_strategy(df: pd.DataFrame, 
                           budget: int,
                           use_half_kelly: bool = True,
                           min_edge: float = 0.05) -> dict:
    """
    Kelly戦略でシミュレーション
    
    Args:
        df: レース結果データ（予測確率、オッズ、着順を含む）
        budget: 1レースあたり予算
        use_half_kelly: Half-Kelly使用
        min_edge: 最小エッジ閾値
        
    Returns:
        シミュレーション結果
    """
    strategy = KellyStrategy(
        use_half_kelly=use_half_kelly,
        min_edge=min_edge
    )
    
    total_bet = 0
    total_return = 0
    race_count = 0
    win_count = 0
    
    # レースIDでグループ化
    if 'race_id' not in df.columns:
        print("Warning: 'race_id' column not found. Using index-based grouping.")
        return {'error': 'No race_id column'}
    
    race_groups = df.groupby('race_id')
    
    for race_id, race_df in race_groups:
        race_count += 1
        
        # 予測確率列を検出
        prob_col = None
        for col in ['pred_prob', 'win_prob', 'prob', 'prediction']:
            if col in race_df.columns:
                prob_col = col
                break
        
        if prob_col is None:
            continue
        
        # オッズ列を検出
        odds_col = None
        for col in ['tan_odds', 'odds', '単勝オッズ']:
            if col in race_df.columns:
                odds_col = col
                break
        
        if odds_col is None:
            continue
        
        # 着順列を検出
        rank_col = None
        for col in ['rank', '着順', 'finish_position']:
            if col in race_df.columns:
                rank_col = col
                break
        
        if rank_col is None:
            continue
        
        race_bet = 0
        race_return = 0
        
        for _, row in race_df.iterrows():
            prob = row.get(prob_col, 0)
            odds = row.get(odds_col, 0)
            rank = row.get(rank_col, 99)
            
            if pd.isna(prob) or pd.isna(odds) or prob <= 0 or odds <= 1:
                continue
            
            # Kelly計算
            edge = strategy.calculate_edge(prob, odds)
            if edge < 1 + min_edge:
                continue
            
            fraction = strategy.calculate_kelly_fraction(prob, odds)
            if fraction <= 0:
                continue
            
            bet_amount = max(100, int(budget * fraction / 100) * 100)
            race_bet += bet_amount
            
            # 単勝: 1着なら払い戻し
            if rank == 1:
                race_return += bet_amount * odds
        
        total_bet += race_bet
        total_return += race_return
        
        if race_return > race_bet:
            win_count += 1
    
    return {
        'strategy': f"Kelly{'(Half)' if use_half_kelly else '(Full)'}",
        'budget': budget,
        'min_edge': min_edge,
        'races': race_count,
        'total_bet': total_bet,
        'total_return': total_return,
        'profit': total_return - total_bet,
        'recovery_rate': total_return / total_bet if total_bet > 0 else 0,
        'win_rate': win_count / race_count if race_count > 0 else 0
    }


def simulate_fixed_bet_strategy(df: pd.DataFrame,
                               budget: int,
                               top_n: int = 1) -> dict:
    """
    固定賭け金戦略（既存戦略の代理）
    予測確率上位N頭に均等配分
    
    Args:
        df: レース結果データ
        budget: 1レースあたり予算
        top_n: 賭ける馬の数
        
    Returns:
        シミュレーション結果
    """
    total_bet = 0
    total_return = 0
    race_count = 0
    win_count = 0
    
    if 'race_id' not in df.columns:
        return {'error': 'No race_id column'}
    
    race_groups = df.groupby('race_id')
    bet_per_horse = budget // top_n
    
    for race_id, race_df in race_groups:
        race_count += 1
        
        # 予測確率列
        prob_col = None
        for col in ['pred_prob', 'win_prob', 'prob']:
            if col in race_df.columns:
                prob_col = col
                break
        
        if prob_col is None:
            continue
        
        # オッズ・着順列
        odds_col = next((c for c in ['tan_odds', 'odds'] if c in race_df.columns), None)
        rank_col = next((c for c in ['rank', '着順'] if c in race_df.columns), None)
        
        if odds_col is None or rank_col is None:
            continue
        
        # 確率上位N頭を選択
        top_horses = race_df.nlargest(top_n, prob_col)
        
        race_bet = 0
        race_return = 0
        
        for _, row in top_horses.iterrows():
            odds = row.get(odds_col, 0)
            rank = row.get(rank_col, 99)
            
            if pd.isna(odds) or odds <= 1:
                continue
            
            race_bet += bet_per_horse
            
            if rank == 1:
                race_return += bet_per_horse * odds
        
        total_bet += race_bet
        total_return += race_return
        
        if race_return > race_bet:
            win_count += 1
    
    return {
        'strategy': f"Fixed(top{top_n})",
        'budget': budget,
        'races': race_count,
        'total_bet': total_bet,
        'total_return': total_return,
        'profit': total_return - total_bet,
        'recovery_rate': total_return / total_bet if total_bet > 0 else 0,
        'win_rate': win_count / race_count if race_count > 0 else 0
    }


def run_backtest(data_path: str = None):
    """
    バックテスト実行
    """
    print("=" * 60)
    print("Kelly Strategy Backtest")
    print("=" * 60)
    
    # データ読み込み
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_pickle(data_path)
    else:
        # デフォルトパス
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        df = load_historical_data(data_dir)
    
    if df.empty:
        print("No data available for backtest.")
        print("Creating synthetic test data for demonstration...")
        
        # デモ用の合成データ作成
        np.random.seed(42)
        n_races = 500
        horses_per_race = 14
        
        data = []
        for race in range(n_races):
            race_id = f"2024{race:06d}"
            probs = np.random.dirichlet(np.ones(horses_per_race))
            winner = np.random.choice(horses_per_race, p=probs)
            
            for horse in range(horses_per_race):
                odds = 1 / probs[horse] * np.random.uniform(0.8, 1.2)
                odds = max(1.1, odds)
                rank = 1 if horse == winner else np.random.randint(2, horses_per_race + 1)
                
                data.append({
                    'race_id': race_id,
                    'horse_number': horse + 1,
                    'pred_prob': probs[horse],
                    'tan_odds': round(odds, 1),
                    'rank': rank
                })
        
        df = pd.DataFrame(data)
        print(f"Created synthetic data: {n_races} races, {len(df)} records")
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 予算パターン
    budgets = [5000, 10000, 20000]
    
    results = []
    
    print("\n" + "=" * 60)
    print("Running Kelly Strategy Tests...")
    print("=" * 60)
    
    for budget in budgets:
        # Kelly (Half)
        result = simulate_kelly_strategy(df, budget, use_half_kelly=True, min_edge=0.05)
        results.append(result)
        print(f"\n{result['strategy']} @ ¥{budget:,}:")
        print(f"  回収率: {result['recovery_rate']*100:.1f}%")
        print(f"  賭け金: ¥{result['total_bet']:,.0f}")
        print(f"  払戻金: ¥{result['total_return']:,.0f}")
        
        # Kelly (Full) - より攻撃的
        result = simulate_kelly_strategy(df, budget, use_half_kelly=False, min_edge=0.10)
        results.append(result)
        print(f"\n{result['strategy']} @ ¥{budget:,}:")
        print(f"  回収率: {result['recovery_rate']*100:.1f}%")
        
        # Fixed Strategy (比較用)
        result = simulate_fixed_bet_strategy(df, budget, top_n=1)
        results.append(result)
        print(f"\n{result['strategy']} @ ¥{budget:,}:")
        print(f"  回収率: {result['recovery_rate']*100:.1f}%")
    
    # 結果をDataFrameに
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("Summary Results")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # レポート保存
    report_path = os.path.join(os.path.dirname(__file__), '..', 'kelly_backtest_report.csv')
    results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\nReport saved to: {report_path}")
    
    return results_df


if __name__ == "__main__":
    run_backtest()
