"""
2026 YTD シミュレーション (Strategy B)
学習期間: 2016-2025 (10年)
テスト期間: 2026/01/01 - 2026/01/25
戦略: EV >= 1.0 AND Score >= 0.4
"""
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def simulate_2026():
    print("=== 2026 YTD Simulation (Strategy B) ===")

    # 1. データロード
    print("[1] Loading Data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 前処理
    # 全データを渡して時系列特徴量を計算させるが、学習には未来データを使わない
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    elif len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
        # df_fullがない場合はresults_dfを使う(indexが合っている前提)
        df_full = results_df
    else:
        # Fallback
        vals = ret
        X, y = vals[0], vals[1]
        df_full = results_df

    # 年情報の取得
    if df_full is not None and 'date' in df_full.columns:
        years = df_full['date'].dt.year.values
        # 着順
        if '着順' in df_full.columns:
            ranks = pd.to_numeric(df_full['着順'], errors='coerce').fillna(0).values
        else:
             ranks = np.zeros(len(df_full))
        # 2026年の日付フィルタ用
        dates = df_full['date'].values
    else:
        print("Error: Date column missing.")
        return

    # データフレーム作成
    sim_master_df = X.copy()
    sim_master_df['target'] = y
    sim_master_df['year'] = years
    sim_master_df['rank_res'] = ranks
    
    if 'odds' not in sim_master_df.columns:
        print("Error: Odds column missing.")
        return
    
    odds_col = sim_master_df['odds'].values

    # 特徴量選択 (オッズ関連除外)
    feature_cols = [c for c in X.columns if c not in ['odds', 'popularity', 'jockey_return_avg', 'target', 'year', 'rank_res']]
    print(f"Features: {len(feature_cols)} (Odds-related excluded)")

    # 2. 期間設定
    # Train: 2016 <= year <= 2025
    # Test : 2026/01/01 <= date <= 2026/01/25 (スクレイピング済みの最新データまで)
    
    train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] <= 2025)
    test_mask = (sim_master_df['year'] == 2026)
    
    # Validation (2025年をValにする)
    real_train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] <= 2024)
    val_mask = (sim_master_df['year'] == 2025)
    
    X_train = sim_master_df.loc[real_train_mask, feature_cols]
    y_train = sim_master_df.loc[real_train_mask, 'target']
    X_val = sim_master_df.loc[val_mask, feature_cols]
    y_val = sim_master_df.loc[val_mask, 'target']
    
    X_test = sim_master_df.loc[test_mask, feature_cols]
    
    print(f"Train: {len(X_train)} (2016-2024)")
    print(f"Val  : {len(X_val)} (2025)")
    print(f"Test : {len(X_test)} (2026)")
    
    if len(X_test) == 0:
        print("No data for 2026 found.")
        return

    # カテゴリカル変換
    for col in ['枠番', '馬番']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    # 3. 学習
    print("\n[2] Training Model...")
    model = HorseRaceModel()
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Val AUC: {metrics['auc']:.4f}")
    
    # 4. 予測
    print("\n[3] Predicting 2026...")
    probs = model.predict(X_test)
    
    current_odds = odds_col[test_mask]
    current_ranks = sim_master_df.loc[test_mask, 'rank_res'].values
    current_years = sim_master_df.loc[test_mask, 'year'].values
    
    # メタデータ取得（日付確認用）
    if df_full is not None:
        current_dates = df_full.loc[test_mask, 'date'].values
    else:
        current_dates = [None] * len(probs)
    
    test_df = pd.DataFrame({
        'date': current_dates,
        'score': probs,
        'odds': current_odds,
        'rank': current_ranks
    })
    
    # オッズなし除外
    invest_df = test_df.dropna(subset=['odds']).copy()
    invest_df['EV'] = invest_df['score'] * invest_df['odds']
    
    # 5. 戦略B シミュレーション
    print("\n[4] Applying Strategy B (EV>=1.0 & Score>=0.4)...")
    
    strat_mask = (invest_df['EV'] >= 1.0) & (invest_df['score'] >= 0.4)
    bet_df = invest_df[strat_mask]
    
    bets = len(bet_df)
    hits = len(bet_df[bet_df['rank'] == 1])
    
    invest = bets * 1000
    ret = bet_df[bet_df['rank'] == 1]['odds'].sum() * 1000 if bets > 0 else 0
    rate = (ret / invest * 100) if invest > 0 else 0
    
    print(f"\n=== Result (2026/01/01 - 2026/01/25) ===")
    print(f"Bets: {bets} / {len(invest_df)} horses")
    print(f"Hits: {hits}")
    print(f"Invest: {invest:,} Yen")
    print(f"Return: {ret:,.0f} Yen")
    print(f"Recovery Rate: {rate:.1f}%")
    
    # 日付別内訳
    print("\n--- Daily Breakdown ---")
    bet_df['date_dt'] = pd.to_datetime(bet_df['date'])
    daily = bet_df.groupby('date_dt').apply(lambda x: pd.Series({
        'bets': len(x),
        'hits': len(x[x['rank'] == 1]),
        'invest': len(x) * 1000,
        'return': x[x['rank'] == 1]['odds'].sum() * 1000
    })).sort_index()
    
    daily['rate'] = daily['return'] / daily['invest'] * 100
    print(daily)
    
    # 結果保存
    bet_df.to_csv('simulation_2026_strategy_B.csv', index=False)
    print("\nSaved detailed log to simulation_2026_strategy_B.csv")

if __name__ == '__main__':
    simulate_2026()
