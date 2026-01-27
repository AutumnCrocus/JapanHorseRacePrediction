"""
Production Model Training Script
2016-2025年の全データを使用して本番用モデルを学習し、必要なアーティファクトを保存する。
保存するファイル:
1. models/production_model.pkl (LightGBM)
2. models/bias_map.pkl (Track Bias Map)
3. models/jockey_stats.pkl (Latest Jockey Stats)

Scale: False (Tree-based model doesn't need scaling, and simplifies inference)
Features: Pure Ability (No odds/popularity)
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def train_production():
    print("=== Training Production Model (2016-2025) ===")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. データロード
    print("[1] Loading Data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 前処理
    # scale=False
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    elif len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
        df_full = results_df
    else:
        # Should not happen with latest preprocessing.py
        print("Error: prepare_training_data returned unexpected values.")
        return

    # データの期間フィルタリング (念のため)
    # df_fullを使って2016-2025に絞る
    if df_full is not None and 'date' in df_full.columns:
        years = df_full['date'].dt.year
        # 学習範囲: 2016-2025 (2025含む)
        # ※ ユーザーのリクエストでは「2026年のシミュレーション」を別でやったが、
        # 本番モデルには2025年までのデータを含めるべき。
        mask = (years >= 2016) & (years <= 2025)
        
        X_train = X.loc[mask].copy()
        y_train = y.loc[mask].copy()
        print(f"Training Data: {len(X_train)} rows (2016-2025)")
    else:
        print("Warning: Date column missing, using all data.")
        X_train = X.copy()
        y_train = y.copy()

    # 特徴量選択 (オッズ関連除外)
    # 注意: bias_mapは既にXに含まれている(waku_bias_rate)が、
    # 予測時に再現するために bias_map 自体も保存する必要がある。
    
    exclude_cols = ['odds', 'popularity', 'jockey_return_avg', 'target', 'year', 'rank_res']
    feature_cols = [c for c in X_train.columns if c not in exclude_cols]
    
    print(f"Features: {len(feature_cols)}")
    print(feature_cols)
    
    X_train = X_train[feature_cols]

    # カテゴリカル変換
    for col in ['枠番', '馬番']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')

    # 学習
    print("\n[2] Training...")
    model = HorseRaceModel()
    # 全データで学習 (Valなし) -> Early Stoppingできないので、適当なround数決めるか、一部をValにするか。
    # ここでは最終モデルなので、2025年をValにしてEarly Stoppingし、その後全データで再学習...はLightGBMでは難しい。
    # 一般的には「Early Stoppingで決めたround数 x 1.1倍」などで全データ学習するが、
    # 簡便のため 2024までTrain, 2025をValとして学習する。
    # (本当の最新直近データ2025年を学習に含めたいが、Valに使わないと過学習が怖い)
    
    # 折衷案: 2025年をValに使ってモデルを作る。2025年のデータも学習されていることになる（Valとしてだが）。
    # LightGBMの仕様上、Valデータは学習には「直接は」使われない（評価に使われる）。
    # しかし、ハイパーパラメータ(num_boost_round)の決定に使われる。
    # Production用としては、全データTrainで num_boost_round を固定で回すのが定石だが、
    # ここでは安全策で Split して学習する。
    
    val_mask = (years == 2025)
    train_mask = (years >= 2016) & (years <= 2024)
    
    X_tr = X.loc[train_mask, feature_cols]
    y_tr = y.loc[train_mask]
    X_val = X.loc[val_mask, feature_cols]
    y_val = y.loc[val_mask]
    
    # カテゴリ変換
    for col in ['枠番', '馬番']:
         X_tr[col] = X_tr[col].astype('category')
         X_val[col] = X_val[col].astype('category')
    
    model.train(X_tr, y_tr, X_val=X_val, y_val=y_val)
    
    # 保存
    print("\n[3] Saving Artifacts...")
    
    # Model
    model.save(os.path.join(MODEL_DIR, 'production_model.pkl'))
    print("Saved production_model.pkl")
    
    # Bias Map
    bias_map.to_pickle(os.path.join(MODEL_DIR, 'bias_map.pkl'))
    print("Saved bias_map.pkl")
    
    # Jockey Stats
    if jockey_stats is not None:
        jockey_stats.to_pickle(os.path.join(MODEL_DIR, 'jockey_stats.pkl'))
        print("Saved jockey_stats.pkl")
        
    print("Done.")

if __name__ == '__main__':
    train_production()
