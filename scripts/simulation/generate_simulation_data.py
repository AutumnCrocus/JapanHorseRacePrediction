"""
Generate Simulation Data Script
Rolling Window Simulation のロジックで予測を行い、詳細なメタデータ（race_id, 馬番, date等）を含む
CSVファイル (rolling_prediction_details_v2.csv) を生成する。
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def generate_data():
    print("=== Generating Simulation Data (2022-2025) ===")

    # 1. 全データ読み込み & 前処理（一括）
    print("[1] Loading & Preprocessing All Data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    # デバッグ用に件数を絞る場合はここで行う
    # results_df = results_df.tail(10000) 
    
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 前処理
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    else:
        print("Error: Value extraction failed.")
        return

    # 年情報の取得
    if df_full is not None and 'date' in df_full.columns:
        years = df_full['date'].dt.year.values
        # 着順も取得
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
    
    # オッズ情報を確保
    if 'odds' in sim_master_df.columns:
        odds_col = sim_master_df['odds'].values
    else:
        print("Error: Odds column missing.")
        return

    # 学習用からオッズを除外
    feature_cols = [c for c in X.columns if c not in ['odds', 'popularity', 'jockey_return_avg', 'target', 'year', 'rank_res', 'original_race_id', 'date']]
    print(f"Features: {len(feature_cols)} (Odds/Meta excluded)")

    # ループ処理
    test_years = [2022, 2023, 2024, 2025]
    all_details = []

    for test_year in test_years:
        print(f"\n--- Processing Test Year: {test_year} ---")
        
        train_mask = (sim_master_df['year'] >= 2016) & (sim_master_df['year'] < test_year)
        test_mask = (sim_master_df['year'] == test_year)
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
            
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # カテゴリカル変換
        for col in ['枠番', '馬番']:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')
                X_test[col] = X_test[col].astype('category')

        # 学習
        print("Training Model...")
        model = HorseRaceModel()
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # 予測
        print("Predicting...")
        probs = model.predict(X_test)
        
        # 保存用データ作成
        current_odds = odds_col[test_mask]
        current_ranks = sim_master_df.loc[test_mask, 'rank_res'].values
        current_years = sim_master_df.loc[test_mask, 'year'].values
        
        # 重要なメタカラムを抽出
        meta_cols = ['original_race_id', 'date', 'venue_id', 'race_num', 'course_len', 'race_type', '馬番', '枠番', 'popularity']
        # 存在するものだけ
        avail_meta = [c for c in meta_cols if c in sim_master_df.columns]
        
        meta_data = sim_master_df.loc[test_mask, avail_meta].copy()
        
        # DataFrame作成
        test_df = pd.DataFrame({
            'year': current_years,
            'score': probs,
            'odds': current_odds,
            'rank': current_ranks
        })
        
        # meta情報を結合
        test_df = pd.concat([test_df.reset_index(drop=True), meta_data.reset_index(drop=True)], axis=1)
        all_details.append(test_df)

    # 詳細データ保存
    if all_details:
        full_details = pd.concat(all_details, ignore_index=True)
        # original_race_id がない場合は警告
        if 'original_race_id' not in full_details.columns:
            print("WARNING: original_race_id not found in output!")
            
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'rolling_prediction_details_v2.csv')
        full_details.to_csv(output_path, index=False)
        print(f"Saved details to {output_path}")

if __name__ == '__main__':
    generate_data()
