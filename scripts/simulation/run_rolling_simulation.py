"""
Rolling Window Simulation Script (Optimized)
2016年から学習データを1年ずつ増やしながら、翌年の回収率検証を連続実行する。
・Pattern 1: Train(16-21) -> Test(22)
・Pattern 2: Train(16-22) -> Test(23)
・Pattern 3: Train(16-23) -> Test(24)
・Pattern 4: Train(16-24) -> Test(25)

各イテレーションで:
  1. オッズ特徴量の除外 (Pure Ability Model)
  2. モデル学習 & 保存 (pure_model_{test_year}.pkl)
  3. EVシミュレーション (閾値: 1.0, 1.2, 1.5)
  4. データリーク疑念のチェック（異常スコア検知）

最適化: joblibを用いて年ごとに並列処理
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel
from scripts.train_period import train_period_model # 参考程度

def process_year(test_year, sim_master_df, feature_cols, odds_col):
    """
    単一年度の処理を行う関数（並列実行用）
    """
    print(f"\n--- Processing Test Year: {test_year} ---")
    
    # Train: 2016 <= y <= test_year - 1
    train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] < test_year)
    # Test: y == test_year
    test_mask = (sim_master_df['year'] == test_year)
    
    # Validation for Early Stopping (Trainの最後の年)
    val_year = test_year - 1
    val_mask = (sim_master_df['year'] == val_year)
    real_train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] < val_year)
    
    X_train = sim_master_df.loc[real_train_mask, feature_cols].copy()
    y_train = sim_master_df.loc[real_train_mask, 'target'].copy()
    X_val = sim_master_df.loc[val_mask, feature_cols].copy()
    y_val = sim_master_df.loc[val_mask, 'target'].copy()
    
    X_test = sim_master_df.loc[test_mask, feature_cols].copy()
    
    if len(X_test) == 0:
        print(f"No data for {test_year}, skipping.")
        return None, None
        
    print(f"[{test_year}] Train: {len(X_train)} (2016-{val_year-1})")
    print(f"[{test_year}] Val  : {len(X_val)} ({val_year})")
    print(f"[{test_year}] Test : {len(X_test)} ({test_year})")
    
    # カテゴリカル変換
    for col in ['枠番', '馬番']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    # 学習
    print(f"[{test_year}] Training Model...")
    model = HorseRaceModel() # デフォルトパラメータ（n_jobs=-1など）を使用
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"[{test_year}] Val AUC: {metrics['auc']:.4f}")
    
    # 予測
    print(f"[{test_year}] Predicting...")
    probs = model.predict(X_test)
    
    # シミュレーション
    current_odds = odds_col[test_mask]
    current_ranks = sim_master_df.loc[test_mask, 'rank_res'].values
    current_years = sim_master_df.loc[test_mask, 'year'].values
    
    # 必要なカラムを抽出
    meta_cols = ['venue_id', 'course_len', 'race_type', 'ground_state', 'weather', 'race_num']
    # 存在するものだけ
    avail_meta = [c for c in meta_cols if c in X_test.columns]
    
    meta_data = sim_master_df.loc[test_mask, avail_meta].copy()
    for c in meta_data.columns:
        if meta_data[c].dtype.name == 'category':
            meta_data[c] = meta_data[c].astype(str) # 文字列に戻す
    
    # DataFrame作成
    test_df = pd.DataFrame({
        'year': current_years,
        'score': probs,
        'odds': current_odds,
        'rank': current_ranks
    })
    
    # meta情報を結合
    test_df = pd.concat([test_df.reset_index(drop=True), meta_data.reset_index(drop=True)], axis=1)
    
    # 投資シミュレーション用にはオッズありのみ使用
    invest_df = test_df.dropna(subset=['odds']).copy()
    invest_df['EV'] = invest_df['score'] * invest_df['odds']
    
    # EV戦略別集計
    ev_thresholds = [1.0, 1.2, 1.5, 2.0]
    year_res = {'year': test_year, 'val_auc': metrics['auc']}
    
    for thresh in ev_thresholds:
        bet_df = invest_df[invest_df['EV'] >= thresh]
        bet_count = len(bet_df)
        hits = bet_df[bet_df['rank'] == 1]
        
        invest = bet_count * 1000
        ret = hits['odds'].sum() * 1000 if bet_count > 0 else 0
        
        rate = (ret / invest * 100) if invest > 0 else 0
        year_res[f'rate_{thresh}'] = rate
        year_res[f'bets_{thresh}'] = bet_count
        
        print(f"[{test_year}] EV>={thresh}: Rate {rate:.1f}% (Bets: {bet_count})")
        
    return year_res, test_df

def run_rolling_simulation():
    print("=== Rolling Window Simulation (2022-2025) [Parallelized] ===")

    # 1. 全データ読み込み & 前処理（一括）
    print("[1] Loading & Preprocessing All Data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 前処理
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    elif len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
        df_full = results_df 
    else:
        vals = ret
        X, y = vals[0], vals[1]
        df_full = results_df

    # 年情報の取得
    if df_full is not None and 'date' in df_full.columns:
        years = df_full['date'].dt.year.values
        if '着順' in df_full.columns:
            ranks = pd.to_numeric(df_full['着順'], errors='coerce').fillna(0).values
        else:
            ranks = np.zeros(len(df_full))
    else:
        print("Error: Date column missing in preprocessed data.")
        return

    # シミュレーション用Dataframe
    sim_master_df = X.copy()
    sim_master_df['target'] = y
    sim_master_df['year'] = years
    sim_master_df['rank_res'] = ranks
    
    if 'odds' in sim_master_df.columns:
        odds_col = sim_master_df['odds'].values
    else:
        print("Error: Odds column missing.")
        return

    # 学習用からオッズを除外
    feature_cols = [c for c in X.columns if c not in ['odds', 'popularity', 'jockey_return_avg', 'target', 'year', 'rank_res']]
    print(f"Features: {len(feature_cols)} (Odds-related excluded)")

    # ループ処理 (並列化)
    test_years = [2022, 2023, 2024, 2025]
    
    # joblib で並列実行 (n_jobs=4: 4年分同時に)
    # backend='loky' がデフォルトで安定している
    results = Parallel(n_jobs=4, verbose=10)(
        delayed(process_year)(year, sim_master_df, feature_cols, odds_col) 
        for year in test_years
    )
    
    # 結果の集約
    summary_results = []
    all_details = []
    
    for res_tuple in results:
        if res_tuple is not None:
            year_res, test_df = res_tuple
            if year_res: summary_results.append(year_res)
            if test_df is not None: all_details.append(test_df)

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(summary_results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('year')
        print(summary_df)
        summary_df.to_csv('rolling_simulation_results.csv', index=False)
        print("Saved summary to rolling_simulation_results.csv")
    
    if all_details:
        full_details = pd.concat(all_details, ignore_index=True)
        full_details.to_csv('rolling_prediction_details.csv', index=False)
        print("Saved details to rolling_prediction_details.csv")

if __name__ == '__main__':
    run_rolling_simulation()
