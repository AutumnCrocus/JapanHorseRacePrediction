"""
2026年1月25日までの成績を集計するスクリプト
simulate_2026_YTD.py の出力csvを読み込み、日付でフィルタリングする。
"""
import pandas as pd
import os

def summarize_jan():
    print("=== 2026 YTD (Jan 1 - Jan 25) Summary ===")
    
    file_path = 'simulation_2026_strategy_B.csv'
    if not os.path.exists(file_path):
        print("Error: Simulation result not found.")
        return
        
    df = pd.read_csv(file_path)
    
    # 日付変換
    # pseudo-date の可能性が高いが、race_id順に並んでいれば
    # 1/1〜1/25の範囲で抽出することで「年初のレース」を特定できると仮定。
    df['date_dt'] = pd.to_datetime(df['date'])
    
    # フィルタリング
    # 2026-01-01 <= date <= 2026-01-25
    start_date = pd.to_datetime('2026-01-01')
    end_date = pd.to_datetime('2026-01-25')
    
    target_df = df[(df['date_dt'] >= start_date) & (df['date_dt'] <= end_date)]
    
    if len(target_df) == 0:
        print("No bets found in the target period.")
        return
        
    print(f"\nPeriod: {start_date.date()} - {end_date.date()}")
    
    # 集計
    bets = len(target_df)
    hits = len(target_df[target_df['rank'] == 1])
    
    invest = bets * 1000
    ret = target_df[target_df['rank'] == 1]['odds'].sum() * 1000
    
    rate = ret / invest * 100 if invest > 0 else 0
    
    print(f"Bets: {bets}")
    print(f"Hits: {hits}")
    print(f"Invest: {invest:,} Yen")
    print(f"Return: {ret:,.0f} Yen")
    print(f"Recovery Rate: {rate:.1f}%")
    
    # 日別レポート
    daily = target_df.groupby('date_dt').apply(lambda x: pd.Series({
        'bets': len(x),
        'hits': len(x[x['rank'] == 1]),
        'invest': len(x) * 1000,
        'return': x[x['rank'] == 1]['odds'].sum() * 1000,
        'rate': (x[x['rank'] == 1]['odds'].sum() / len(x) * 100) if len(x) > 0 else 0
    })).sort_index()
    
    print("\n--- Daily Stats ---")
    print(daily[['bets', 'hits', 'rate']])

if __name__ == '__main__':
    summarize_jan()
