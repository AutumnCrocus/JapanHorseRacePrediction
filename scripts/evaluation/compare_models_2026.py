
"""
モデル比較評価スクリプト (2025 vs 2026)
- 対象: 2026年1月〜2月のレース
- 比較モデル:
    - experiment_model_2025.pkl (既存)
    - experiment_model_2026.pkl (新規)
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.data_loader import load_results, load_shutuba
from modules.preprocessing import DataProcessor, FeatureEngineer

def evaluate():
    print("=== Model Comparison (2025 vs 2026) ===")
    
    # 1. モデルロード
    path_2025 = os.path.join(MODEL_DIR, "experiment_model_2025.pkl")
    path_2026 = os.path.join(MODEL_DIR, "experiment_model_2026.pkl")
    
    model_2025 = HorseRaceModel()
    try:
        model_2025.load(path_2025)
    except:
        print(f"Model 2025 not found at {path_2025}")
        return

    model_2026 = HorseRaceModel()
    try:
        model_2026.load(path_2026)
    except:
        print(f"Model 2026 not found at {path_2026}. Run training first.")
        # Continue if only checking 2025 performance? No, comparison needs both.
        return

    # 2. 評価データロード (2026年)
    print("Loading 2026 data...")
    try:
        results_2026 = load_results(2026, 2026)
        if results_2026.empty:
            print("No results found for 2026. Please scrape data first.")
            return
    except Exception as e:
        print(f"Error loading 2026 data: {e}")
        return

    print(f"Loaded {len(results_2026)} races for 2026.")
    
    # 3. 前処理
    print("Preprocessing...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)

    processor = DataProcessor()
    df_proc = processor.process_results(results_2026)
    
    engineer = FeatureEngineer()
    
    # Filter HR
    active_ids = df_proc['horse_id'].unique()
    hr_filtered = hr[hr.index.isin(active_ids)].copy()
    
    df_proc = engineer.add_horse_history_features(df_proc, hr_filtered)
    df_proc = engineer.add_course_suitability_features(df_proc, hr_filtered)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # Target
    if '着順' in df_proc.columns:
        df_proc['target'] = df_proc['着順'].apply(lambda x: 1 if x <= 3 else 0)
    elif 'rank_num' in df_proc.columns:
        df_proc['target'] = df_proc['rank_num'].apply(lambda x: 1 if x <= 3 else 0)
        
    y_true = df_proc['target']
    
    # 4. 予測と評価
    print("Evaluating...")
    
    def get_metrics(model, X, y):
        # 必要な特徴量のみ抽出（欠損は0埋め）
        cols = [c for c in model.feature_names if c in X.columns]
        X_sub = X[cols].copy()
        for c in model.feature_names:
            if c not in X_sub.columns: X_sub[c] = 0
            
        X_sub = X_sub.fillna(0) # Simple fill
        
        y_prob = model.predict(X_sub)
        y_pred = (y_prob > 0.5).astype(int)
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        return acc, auc, y_prob

    # 2025 Model
    acc_25, auc_25, prob_25 = get_metrics(model_2025, df_proc, y_true)
    print(f"Model 2025: Accuracy={acc_25:.4f}, AUC={auc_25:.4f}")
    
    # 2026 Model
    acc_26, auc_26, prob_26 = get_metrics(model_2026, df_proc, y_true)
    print(f"Model 2026: Accuracy={acc_26:.4f}, AUC={auc_26:.4f}")
    
    # 5. 回収率シミュレーション (簡易)
    # 単勝回収率などを計算したいが、データに単勝オッズがあるか？
    # results_2026にはあるはず。
    # ここでは詳細シミュレーションは割愛し、精度のみ比較。
    
    diff_acc = acc_26 - acc_25
    diff_auc = auc_26 - auc_25
    
    print("-" * 30)
    print(f"Improvement: Accuracy {diff_acc:+.4f}, AUC {diff_auc:+.4f}")

if __name__ == "__main__":
    evaluate()
