"""
フィルタリング条件分析スクリプト
Rolling Windowシミュレーションで保存された `rolling_prediction_details.csv` を読み込み、
条件別の回収率、的中率、購入頭数、ROIを計算する。

分析項目:
1. 過剰人気分析 (Predict Rank vs Actual Pop)
   - 人気はあるが、モデル評価が低い馬 (危険な人気馬)
   - 人気はないが、モデル評価が高い馬 (美味しい穴馬)
2. ファクター分析
   - 競馬場 (venue_id)
   - 距離区分 (Sprint, Mile, Intermediate, Long)
   - トラック (芝, ダート, 障害)
   - 馬場状態 (良, 稍重, 重, 不良)
3. 複合条件探索
   - 「EV >= 1.0 かつ ダート」などの条件でのパフォーマンス
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_filtering():
    print("=== Filtering Analysis ===")
    
    file_path = 'rolling_prediction_details.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run rolling simulation first.")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='shift_jis') # Fallback
        
    print(f"Loaded {len(df)} rows.")
    
    # 前処理
    df['EV'] = df['score'] * df['odds']
    df['is_win'] = (df['rank'] == 1)
    
    # 距離区分
    def classify_distance(d):
        try:
            d = int(d)
            if d < 1400: return 'Sprint'
            elif d < 1800: return 'Mile'
            elif d < 2200: return 'Intermediate'
            else: return 'Long'
        except: return 'Unknown'
        
    df['dist_type'] = df['course_len'].apply(classify_distance)
    
    # 基本統計量
    print("\n--- Basic Stats (EV >= 1.0) ---")
    base_df = df[df['EV'] >= 1.0]
    base_bets = len(base_df)
    base_return = base_df[base_df['is_win']]['odds'].sum()
    base_rate = base_return / base_bets * 100 if base_bets > 0 else 0
    print(f"Base Recovery Rate: {base_rate:.1f}% (Bets: {base_bets})")

    # 1. 競馬場別
    print("\n--- Venue Analysis (EV >= 1.0) ---")
    venue_grp = base_df.groupby('venue_id').apply(lambda x: pd.Series({
        'bets': len(x),
        'win_rate': x['is_win'].mean() * 100,
        'return_rate': x[x['is_win']]['odds'].sum() / len(x) * 100 if len(x) > 0 else 0
    })).sort_values('return_rate', ascending=False)
    print(venue_grp)
    
    # 2. トラック・距離別
    print("\n--- Track/Distance Analysis (EV >= 1.0) ---")
    td_grp = base_df.groupby(['race_type', 'dist_type']).apply(lambda x: pd.Series({
        'bets': len(x),
        'win_rate': x['is_win'].mean() * 100,
        'return_rate': x[x['is_win']]['odds'].sum() / len(x) * 100 if len(x) > 0 else 0
    })).sort_values('return_rate', ascending=False)
    print(td_grp)

    # 3. 過剰人気分析 (人気ランク vs 予測ランク)
    # 人気ランクがないので、oddsでランク付けが必要（レースごと）
    # しかし、ここには全馬データはないかもしれない（詳細CSVに全馬保存している前提）
    # run_rolling_simulation.py の修正で全馬保存するようにしていればOK。
    # していなければ、データ不足。
    # -> 前回の修正で test_df 全体を保存しているはず。
    
    # レースIDがない... yearとrace_numとvenue_idでグルーピング必要だが、日付がないと一意にならないかも？
    # -> rolling_simulation_details には race_id や date がない。
    # -> 修正が必要か？
    # -> run_rolling_simulation.py で sim_master_df から必要なカラムを落としてしまっている。
    # -> しかし、odds があるので単勝人気ランクは作れる（疑似的）。
    
    # 簡易的に、オッズで区分けして分析
    print("\n--- Odds Range Analysis (Odds Range) ---")
    bins = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 9999.0]
    labels = ['1-3', '3-5', '5-10', '10-20', '20-50', '50-100', '100+']
    base_df_odds = base_df.copy()
    base_df_odds['odds_bin'] = pd.cut(base_df_odds['odds'], bins=bins, labels=labels)
    
    odds_grp = base_df_odds.groupby('odds_bin', observed=True).apply(lambda x: pd.Series({
        'bets': len(x),
        'win_rate': x['is_win'].mean() * 100,
        'return_rate': x[x['is_win']]['odds'].sum() / len(x) * 100 if len(x) > 0 else 0
    }))
    print(odds_grp)
    
    # 4. Score別分析
    print("\n--- Model Score Analysis (EV >= 1.0) ---")
    # Scoreが高くてもオッズが低すぎるとEV<1.0になるので、EVフィルタ前で見るべき？
    # いや、EV戦略の改善なのでEV>=1.0の中で、さらにScoreが高い方が良いかを見る。
    
    score_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    base_df_score = base_df.copy()
    base_df_score['score_bin'] = pd.cut(base_df_score['score'], bins=score_bins)
    
    score_grp = base_df_score.groupby('score_bin', observed=True).apply(lambda x: pd.Series({
        'bets': len(x),
        'return_rate': x[x['is_win']]['odds'].sum() / len(x) * 100 if len(x) > 0 else 0
    }))
    print(score_grp)

    # 推奨条件の提案 (全探索)
    print("\n=== Finding Best Conditions ===")
    best_rate = 0
    best_cond = ""
    
    # 組み合わせ候補
    venue_list = df['venue_id'].unique()
    race_types = df['race_type'].unique()
    dist_types = df['dist_type'].unique()
    
    # シンプルなグリッドサーチ
    # 例: ダートダート短距離、ローカル競馬場など
    # ここでは手動でいくつか有望なものを試す形にする（全探索は重い）
    
    # 1. ダート vs 芝
    for rt in ['ダ', '芝']:
        cond_df = base_df[base_df['race_type'] == rt]
        if len(cond_df) > 100:
            rate = cond_df[cond_df['is_win']]['odds'].sum() / len(cond_df) * 100
            print(f"Type: {rt}, Rate: {rate:.1f}%, Bets: {len(cond_df)}")

    # 2. オッズ5-20倍 (中穴)
    mid_odds_df = base_df[(base_df['odds'] >= 5.0) & (base_df['odds'] < 20.0)]
    rate = mid_odds_df[mid_odds_df['is_win']]['odds'].sum() / len(mid_odds_df) * 100
    print(f"Odds 5-20: Rate: {rate:.1f}%, Bets: {len(mid_odds_df)}")
    
    # 3. 高スコア (Score >= 0.5)
    high_score_df = base_df[base_df['score'] >= 0.5]
    if len(high_score_df) > 0:
        rate = high_score_df[high_score_df['is_win']]['odds'].sum() / len(high_score_df) * 100
        print(f"Score >= 0.5: Rate: {rate:.1f}%, Bets: {len(high_score_df)}")
    
if __name__ == '__main__':
    analyze_filtering()
