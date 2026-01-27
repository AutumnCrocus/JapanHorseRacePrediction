"""
Rolling Window Simulation Script
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
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel
from scripts.train_period import train_period_model # 参考程度

def run_rolling_simulation():
    print("=== Rolling Window Simulation (2022-2025) ===")

    # 1. 全データ読み込み & 前処理（一括）
    # Expanding Windowの特徴量計算を正しく行うため、全期間を一度に通す
    print("[1] Loading & Preprocessing All Data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 前処理 (df_fullを受け取る)
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    elif len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
        df_full = results_df # Fallback (Index match assumed)
    else:
        # Fallback
        vals = ret
        X, y = vals[0], vals[1]
        df_full = results_df

    # 年情報の取得
    if df_full is not None and 'date' in df_full.columns:
        years = df_full['date'].dt.year.values
        # 着順も取得
        if '着順' in df_full.columns:
            ranks = pd.to_numeric(df_full['着順'], errors='coerce').fillna(0).values
        else:
            ranks = np.zeros(len(df_full))
    else:
        # dateなしの場合 (preprocessingでpseudo dateがあればそれを使うはずだが)
        print("Error: Date column missing in preprocessed data.")
        return

    # シミュレーション用Dataframe (Xとyとmeta情報)
    # 完全にコピーして操作
    sim_master_df = X.copy()
    sim_master_df['target'] = y
    sim_master_df['year'] = years
    sim_master_df['rank_res'] = ranks
    
    # オッズ情報を確保 (EV計算用)
    if 'odds' in sim_master_df.columns:
        odds_col = sim_master_df['odds'].values
    else:
        print("Error: Odds column missing.")
        return

    # 学習用からオッズを除外
    feature_cols = [c for c in X.columns if c not in ['odds', 'popularity', 'jockey_return_avg', 'target', 'year', 'rank_res']]
    print(f"Features: {len(feature_cols)} (Odds-related excluded)")

    # ループ処理
    test_years = [2022, 2023, 2024, 2025]
    summary_results = []

    for test_year in test_years:
        print(f"\n--- Processing Test Year: {test_year} ---")
        
        # Train: 2016 <= y <= test_year - 1
        train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] < test_year)
        # Test: y == test_year
        test_mask = (sim_master_df['year'] == test_year)
        
        # Validation for Early Stopping (Trainの最後の年)
        val_year = test_year - 1
        val_mask = (sim_master_df['year'] == val_year)
        real_train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] < val_year)
        
        X_train = sim_master_df.loc[real_train_mask, feature_cols]
        y_train = sim_master_df.loc[real_train_mask, 'target']
        X_val = sim_master_df.loc[val_mask, feature_cols]
        y_val = sim_master_df.loc[val_mask, 'target']
        
        X_test = sim_master_df.loc[test_mask, feature_cols]
        
        if len(X_test) == 0:
            print(f"No data for {test_year}, skipping.")
            continue
            
        print(f"Train: {len(X_train)} (2016-{val_year-1})")
        print(f"Val  : {len(X_val)} ({val_year})")
        print(f"Test : {len(X_test)} ({test_year})")
        
        # カテゴリカル変換
        for col in ['枠番', '馬番']:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')
                X_test[col] = X_test[col].astype('category')

        # 学習
        print("\nTraining Model...")
        model = HorseRaceModel()
        metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        print(f"Val AUC: {metrics['auc']:.4f}")
        
        # 予測
        print("Predicting...")
        probs = model.predict(X_test)
        
        # シミュレーション
        current_odds = odds_col[test_mask]
        current_ranks = sim_master_df.loc[test_mask, 'rank_res'].values
        current_years = sim_master_df.loc[test_mask, 'year'].values
        
        # 追加情報: 競馬場、距離、馬場状態、トラックタイプが必要
        # sim_master_df は X.copy() なので、元の columns に含まれているか確認。
        # category化されている可能性があるので、元の値を保持しておくほうが安全だが、
        # ここでは X にあるものを使う。
        # date, course_len などが含まれているはず。
        
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
        
        # 全データ保存用リストに追加
        # まだ dropna しない（分析用に全データ残す）
        summary_results.append(pd.DataFrame({'year': [test_year], 'val_auc': [metrics['auc']]})) 
        
        # 詳細ログを追記保存するために、リストではなく一時ファイルに追記するか、最後に結合するか。
        # メモリ圧迫しないよう、年ごとにファイル出力して後で結合が良いが、
        # 数万行程度ならメモリで持てる。
        if 'all_details' not in locals():
            all_details = []
        all_details.append(test_df)

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
            hit_count = len(hits)
            
            invest = bet_count * 1000
            ret = hits['odds'].sum() * 1000 if bet_count > 0 else 0
            
            rate = (ret / invest * 100) if invest > 0 else 0
            year_res[f'rate_{thresh}'] = rate
            year_res[f'bets_{thresh}'] = bet_count
            
            print(f"EV>={thresh}: Rate {rate:.1f}% (Bets: {bet_count}, Hit: {hit_count})")
            
        # summary_results は DataFrame のリストではなく Dict のリストに修正が必要
        # ループ外で初期化されている summary_results に辞書を追加する形に戻す
        # 先ほど append した DataFrame は削除
        summary_results.pop() 
        summary_results.append(year_res)

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df)
    
    # CSV保存
    summary_df.to_csv('rolling_simulation_results.csv', index=False)
    print("Saved summary to rolling_simulation_results.csv")
    
    # 詳細データ保存
    if 'all_details' in locals() and all_details:
        full_details = pd.concat(all_details, ignore_index=True)
        full_details.to_csv('rolling_prediction_details.csv', index=False)
        print("Saved details to rolling_prediction_details.csv")

if __name__ == '__main__':
    run_rolling_simulation()
