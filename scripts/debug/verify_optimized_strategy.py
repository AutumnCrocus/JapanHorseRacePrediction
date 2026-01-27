"""
最適化戦略検証スクリプト
analyze_filtering.py の結果に基づき、以下の戦略の年次パフォーマンスを検証する。

Strategy A: EV >= 1.0 AND Score >= 0.2 (Broad)
Strategy B: EV >= 1.0 AND Score >= 0.4 (Strict)

入力: rolling_prediction_details.csv (2022-2025)
出力: 年ごとの回収率、的中率、ROI
"""
import pandas as pd
import os

def verify_strategy():
    print("=== Optimized Strategy Verification (2022-2025) ===")
    
    file_path = 'rolling_prediction_details.csv'
    if not os.path.exists(file_path):
        print("Error: detailed log not found.")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='shift_jis')

    # 基本前処理
    df['EV'] = df['score'] * df['odds']
    df['is_win'] = (df['rank'] == 1)
    
    # 戦略定義
    strategies = {
        'Strategy A (Score>=0.2)': lambda d: (d['EV'] >= 1.0) & (d['score'] >= 0.2),
        'Strategy B (Score>=0.4)': lambda d: (d['EV'] >= 1.0) & (d['score'] >= 0.4)
    }
    
    results = []
    
    years = sorted(df['year'].unique())
    for strat_name, func in strategies.items():
        print(f"\n--- {strat_name} ---")
        strat_res = []
        for year in years:
            ydf = df[df['year'] == year]
            mask = func(ydf)
            bet_df = ydf[mask]
            
            bets = len(bet_df)
            hits = bet_df['is_win'].sum()
            invest = bets * 100
            ret = bet_df[bet_df['is_win']]['odds'].sum() * 100
            rate = ret / invest * 100 if invest > 0 else 0
            
            print(f"{year}: Rate {rate:.1f}% (Bets: {bets}, Hits: {hits})")
            
            strat_res.append({
                'strategy': strat_name,
                'year': year,
                'rate': rate,
                'bets': bets,
                'hits': hits
            })
            
        # Total
        mask = func(df)
        total_bets = len(df[mask])
        total_ret = df[mask][df[mask]['is_win']]['odds'].sum() * 100
        total_rate = total_ret / (total_bets * 100) * 100 if total_bets > 0 else 0
        print(f"TOTAL: Rate {total_rate:.1f}% (Bets: {total_bets})")
        
        results.extend(strat_res)

    # 結果保存
    res_df = pd.DataFrame(results)
    res_df.to_csv('optimized_strategy_results.csv', index=False)
    print("\nSaved to optimized_strategy_results.csv")

if __name__ == '__main__':
    verify_strategy()
