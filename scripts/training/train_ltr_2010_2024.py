
"""
ランキング学習 (LambdaMART) ヒストリカルモデル作成スクリプト (2010-2024)
- 目的: 2010-2024年のデータで学習し、2025年以降のシミュレーションに使用する。
- アルゴリズム: LightGBM (lambdarank objective)
- 評価指標: NDCG@1, NDCG@3, NDCG@5
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import DataProcessor, FeatureEngineer

# ============= 設定 =============
TRAIN_START = '2010-01-01'
TRAIN_END = '2024-12-31'
# 学習データの中で検証用データを確保する (2024年データを検証用とする)
VAL_START = '2024-01-01'
VAL_END = '2024-12-31'

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "historical_ltr_2010_2024")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
# ===============================

class StandaloneRankingModel:
    """ランキング学習専用の独立モデルクラス"""
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.model = None
        self.feature_names = None

    def train(self, X_train, y_train, group_train, X_val, y_val, group_val, early_stopping_rounds=50):
        """
        ランキング学習の実行
        group: 各グループ（レース）内のサンプル数リスト
        """
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        self.feature_names = X_train.columns.tolist()

    def predict(self, X):
        """ランキングスコア（相対的な強さ）を予測"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X[self.feature_names])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.params
            }, f)

def prepare_ranking_labels(df):
    """着順をランキング学習用の関連度スコアに変換"""
    # 1着: 3, 2着: 2, 3着: 1, その他: 0 (より高い値が「良い」)
    if 'rank' in df.columns:
        return df['rank'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else (1 if x == 3 else 0)))
    return df['着順'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else (1 if x == 3 else 0)))

def main():
    print("=== ヒストリカルランキング学習 (2010-2024) 開始 ===")
    
    # 1. データ読み込み
    print("データをロード中...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)

    # データ整形
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    # カラム重複除去
    results = results.loc[:, ~results.columns.duplicated()]

    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    # 学習対象期間でフィルタリング (2010-2024)
    results = results[(results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)].copy()
    print(f"対象期間のデータ数: {len(results)}件 ({results['date'].min()} ~ {results['date'].max()})")

    # 2. 特徴量エンジニアリング
    print("特徴量生成中...")
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # サンプルモードのチェック
    is_sample = "--sample" in sys.argv
    if is_sample:
        print("!!! サンプルモード (10%) で実行中 !!!")
        unique_races = results['race_id'].unique()
        sampled_races = np.random.choice(unique_races, size=int(len(unique_races) * 0.1), replace=False)
        results = results[results['race_id'].isin(sampled_races)].copy()
        print(f"サンプリング後のレース数: {len(sampled_races):,}")

    # 特徴量生成の実行
    df_proc = processor.process_results(results.copy())
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # 3. データセット構築 (ランキング学習用)
    print("ランキング学習用データセット構築中...")
    
    # ターゲット: 関連度スコア
    df_proc['relevance'] = prepare_ranking_labels(df_proc)
    
    # 期間指定によるTrain/Val分割 (2024年を検証用とする)
    # Train: 2010-2023, Val: 2024
    train_split_date = '2023-12-31'
    
    df_train = df_proc[df_proc['date'] <= train_split_date].copy()
    df_val = df_proc[(df_proc['date'] >= VAL_START) & (df_proc['date'] <= VAL_END)].copy()
    
    # 学習対象カラムの抽出 (数値のみ)
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 'relevance',
        'time', '着差', '通過', '上り', '単勝', '人気', 'horse_name', 'jockey', 
        'trainer', 'owner', 'gender', 'original_race_id', 'タイム', 'タイム秒',
        '着 順', '不正', '失格', '中止', '取消', '除外', 'running_style',
        '体重', '体重変化', '馬体重', '単 勝', '人 気', '賞金', '賞金（万円）',
        '付加賞（万円）', 'rank_num', 'is_win', 'is_place', 'last_3f_num',
        'odds', 'popularity', 'return'
    ]
    # df_train基準で特徴量を決定
    features = [c for c in df_train.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]
    
    # 特徴量の確認
    print(f"使用特徴量 ({len(features)}): {', '.join(features)}")

    # ランキング学習にはグループ化（レースごとのデータ）が必要
    def get_groups(df):
        return df.groupby('race_id', sort=False).size().tolist()

    train_groups = get_groups(df_train)
    val_groups = get_groups(df_val)
    
    X_train = df_train[features]
    y_train = df_train['relevance']
    X_val = df_val[features]
    y_val = df_val['relevance']
    
    print(f"学習: {len(X_train)}件 ({len(train_groups)}レース) - ~2023")
    print(f"検証: {len(X_val)}件 ({len(val_groups)}レース) - 2024")
    
    # 4. 学習
    model = StandaloneRankingModel()
    model.train(X_train, y_train, train_groups, X_val, y_val, val_groups)
    
    # 重要度の表示
    importances = pd.Series(model.model.feature_importance(), index=features).sort_values(ascending=False)
    print("\n=== 特徴量重要度 (TOP 10) ===")
    print(importances.head(10))
    
    # 5. 保存
    model_path = os.path.join(MODEL_OUTPUT_DIR, "ranking_model.pkl")
    model.save(model_path)
    print(f"モデルを保存しました: {model_path}")
    
    # プロセッサ等も併せて保存（独立性を保つため）
    with open(os.path.join(MODEL_OUTPUT_DIR, "processor.pkl"), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(MODEL_OUTPUT_DIR, "engineer.pkl"), 'wb') as f:
        pickle.dump(engineer, f)
        
    print("=== 学習完了 ===")

if __name__ == "__main__":
    main()
