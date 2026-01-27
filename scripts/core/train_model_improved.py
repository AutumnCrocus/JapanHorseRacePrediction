"""
モデル再学習スクリプト (バイアス改善・リーク対策・検証強化版)
・枠順バイアス特徴量 (waku_bias_rate) を導入
・騎手特徴量のリーク対策 (時系列集計)
・時系列データ分割によるモデル検証
・BiasMap, JockeyStats の保存
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

def train_improved_model():
    print("=== モデル再学習 (バイアス改善・リーク対策版) ===")
    
    # 1. データ読み込み
    print("[1/6] データ読み込み...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} が見つかりません。")
        return

    results_df = pd.read_pickle(results_path)
    
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    print(f"Results: {len(results_df)} rows")
    
    # 2. 前処理 & 特徴量生成
    print("\n[2/6] 前処理 & 特徴量生成...")
    # prepare_training_data は (X, y, processor, engineer, bias_map, jockey_stats) を返す
    # 戻り値のアンパックを柔軟に
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
    elif len(ret) == 5:
        X, y, processor, engineer, bias_map = ret
        jockey_stats = None
        print("Warning: jockey_stats not returned (old processing logic?)")
    else:
        # Fallback
        vals = ret
        X, y, processor, engineer = vals[0], vals[1], vals[2], vals[3]
        bias_map = None
        jockey_stats = None
        print("Warning: bias_map/jockey_stats not returned")

    # 枠番、馬番をカテゴリカルに変換
    for col in ['枠番', '馬番']:
        if col in X.columns:
            X[col] = X[col].astype('category')

    print(f"Features: {X.shape[1]}")
    
    # 3. データ分割 (時系列)
    print("\n[3/6] データ分割 (Time Series Split)...")
    # 単純に後ろの20%を検証用にする (レースID等のソートが前提だが、概ね時系列と仮定)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 4. 学習
    print("\n[4/6] モデル学習 (LightGBM)...")
    model = HorseRaceModel()
    
    # 学習実行 (検証データを明示的に渡す)
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 特徴量重要度確認
    if hasattr(model.model, 'feature_importance'):
        imp = model.model.feature_importance(importance_type='gain')
        names = model.model.feature_name()
        imp_df = pd.DataFrame({'feature': names, 'gain': imp}).sort_values('gain', ascending=False)
        print("\nTop 15 Feature Importance:")
        print(imp_df.head(15))
        
        # チェック対象
        check_feats = ['waku_bias_rate', 'jockey_avg_rank', 'jockey_win_rate', 'jockey_return_avg', 'odds', 'popularity', '枠番', '馬番']
        print("\nBias Check:")
        for f in check_feats:
            if f in imp_df['feature'].values:
                row = imp_df[imp_df['feature']==f].iloc[0]
                print(f"{f}: Rank {imp_df.index.get_loc(row.name)+1}, Gain {row['gain']:.2f}")
            else:
                print(f"{f}: Not used")

    # 5. 保存
    print("\n[5/6] モデル・データ保存...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model.save(os.path.join(MODEL_DIR, 'horse_race_model.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
        
    if bias_map is not None:
        bias_map.to_pickle(os.path.join(MODEL_DIR, 'bias_map.pkl'))
        print("bias_map.pkl saved.")
        
    if jockey_stats is not None:
        jockey_stats.to_pickle(os.path.join(MODEL_DIR, 'jockey_stats.pkl'))
        print("jockey_stats.pkl saved.")

    print("\n[6/6] 完了")
    print("新しいモデルと統計データが保存されました。")

if __name__ == '__main__':
    train_improved_model()
