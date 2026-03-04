"""
ランキング学習 (LambdaMART) 最新版モデル作成スクリプト (2010-2026)

- 目的: 2025-2026年の最新データを含む形でLTRモデルを再学習し、
        Concept Drift（環境変化への未追従）に対応する。
- アルゴリズム: LightGBM (lambdarank objective)
- 学習期間: 2010/01/01 ~ 2026/02/28
- 評価指標: NDCG@1, NDCG@3, NDCG@5
- 出力先: models/historical_ltr_2010_2026/
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
TRAIN_END = '2026-02-28'   # 最新取得済みデータまで延長
VAL_START = '2025-10-01'   # 直近5ヶ月を検証用とする
VAL_END = '2026-02-28'

MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, "historical_ltr_2010_2026")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
# ===============================


class StandaloneRankingModel:
    """ランキング学習専用の独立モデルクラス。"""

    def __init__(self, params: dict = None) -> None:
        """初期化。"""
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
            'random_state': 42,
        }
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: list,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        group_val: list,
        early_stopping_rounds: int = 50,
    ) -> None:
        """ランキング学習の実行。"""
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
                lgb.log_evaluation(period=100),
            ],
        )
        self.feature_names = X_train.columns.tolist()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ランキングスコア（相対的な強さ）を予測。"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X[self.feature_names])

    def save(self, path: str) -> None:
        """モデルをpickle保存。"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.params,
            }, f)


def prepare_ranking_labels(df: pd.DataFrame) -> pd.Series:
    """着順をランキング学習用の関連度スコアに変換。1着=3, 2着=2, 3着=1, その他=0。"""
    col = 'rank' if 'rank' in df.columns else '着順'
    return df[col].apply(lambda x: 3 if x == 1 else (2 if x == 2 else (1 if x == 3 else 0)))


def get_groups(df: pd.DataFrame) -> list:
    """グループ（レースごとのデータ）サイズリストを返す。"""
    return df.groupby('race_id', sort=False).size().tolist()


def main() -> None:
    """メイン関数。"""
    print(f"=== LTR ヒストリカル再学習 ({TRAIN_START} ~ {TRAIN_END}) 開始 ===")
    print(f"  検証期間: {VAL_START} ~ {VAL_END}")
    print(f"  出力先: {MODEL_OUTPUT_DIR}")
    print(f"  実行日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

    # 1. データ読み込み (年別分割pickleを統合)
    print("\n[1/5] データをロード中...")
    results_list = []
    results_yearly_dir = os.path.join(RAW_DATA_DIR, "results")
    if os.path.isdir(results_yearly_dir):
        for year in range(2010, 2027):
            year_path = os.path.join(results_yearly_dir, str(year), f"results_{year}.pkl")
            if os.path.exists(year_path):
                with open(year_path, 'rb') as f:
                    df_year = pickle.load(f)
                results_list.append(df_year)
                print(f"  {year}: {len(df_year):,} 行")
        if results_list:
            results = pd.concat(results_list)
        else:
            raise FileNotFoundError("年別pickleが見つかりません")
    else:
        # フォールバック: 従来の単一ファイル
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
    results = results.loc[:, ~results.columns.duplicated()]

    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(
            results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce'
        )
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    # 学習対象期間フィルタリング
    results = results[
        (results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)
    ].copy()
    print(f"  対象期間のデータ数: {len(results):,} 件 ({results['date'].min()} ~ {results['date'].max()})")

    # サンプルモード
    is_sample = "--sample" in sys.argv
    if is_sample:
        print("!!! サンプルモード (10%) で実行中 !!!")
        sampled_races = np.random.choice(
            results['race_id'].unique(),
            size=max(1, int(results['race_id'].nunique() * 0.1)),
            replace=False
        )
        results = results[results['race_id'].isin(sampled_races)].copy()
        print(f"  サンプリング後: {len(results):,} 件")

    # 2. 特徴量エンジニアリング
    print("\n[2/5] 特徴量生成中...")
    processor = DataProcessor()
    engineer = FeatureEngineer()

    df_proc = processor.process_results(results.copy())
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)

    # 3. データセット構築
    print("\n[3/5] ランキング学習用データセット構築中...")
    df_proc['relevance'] = prepare_ranking_labels(df_proc)

    # date カラムを確保
    if isinstance(df_proc.get('date'), pd.DataFrame):
        df_proc['date'] = df_proc['date'].iloc[:, 0]

    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 'relevance',
        'time', '着差', '通過', '上り', '単勝', '人気', 'horse_name', 'jockey',
        'trainer', 'owner', 'gender', 'original_race_id', 'タイム', 'タイム秒',
        '着 順', '不正', '失格', '中止', '取消', '除外', 'running_style',
        '体重', '体重変化', '馬体重', '単 勝', '人 気', '賞金', '賞金（万円）',
        '付加賞（万円）', 'rank_num', 'is_win', 'is_place', 'last_3f_num',
        'odds', 'popularity', 'return',
    ]

    # Train/Val分割: TRAIN_START~VAL_START前まで学習, VAL_START~VAL_ENDで検証
    train_split_date = (
        pd.Timestamp(VAL_START) - pd.Timedelta(days=1)
    ).strftime('%Y-%m-%d')

    df_train = df_proc[df_proc['date'] <= train_split_date].copy()
    df_val = df_proc[
        (df_proc['date'] >= VAL_START) & (df_proc['date'] <= VAL_END)
    ].copy()

    features = [
        c for c in df_train.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    print(f"  使用特徴量数: {len(features)}")
    print(f"  学習: {len(df_train):,} 件 ({df_train['race_id'].nunique():,} レース) ~ {train_split_date}")
    print(f"  検証: {len(df_val):,} 件 ({df_val['race_id'].nunique():,} レース) {VAL_START}~{VAL_END}")

    train_groups = get_groups(df_train)
    val_groups = get_groups(df_val)

    X_train = df_train[features].fillna(0)
    y_train = df_train['relevance']
    X_val = df_val[features].fillna(0)
    y_val = df_val['relevance']

    # 4. 学習
    print("\n[4/5] LambdaMART モデルを学習中...")
    model = StandaloneRankingModel()
    model.train(X_train, y_train, train_groups, X_val, y_val, val_groups)

    # 特徴量重要度
    importances = pd.Series(
        model.model.feature_importance(), index=features
    ).sort_values(ascending=False)
    print("\n=== 特徴量重要度 (TOP 10) ===")
    print(importances.head(10).to_string())

    # 5. 保存
    print(f"\n[5/5] モデルを保存中: {MODEL_OUTPUT_DIR}")
    model.save(os.path.join(MODEL_OUTPUT_DIR, "ranking_model.pkl"))
    with open(os.path.join(MODEL_OUTPUT_DIR, "processor.pkl"), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(MODEL_OUTPUT_DIR, "engineer.pkl"), 'wb') as f:
        pickle.dump(engineer, f)

    # メタ情報保存
    import json
    meta = {
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_end': VAL_END,
        'num_features': len(features),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'created_at': datetime.now().isoformat(),
    }
    with open(os.path.join(MODEL_OUTPUT_DIR, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("[DONE] LTR モデル再学習完了")
    print(f"   出力先: {MODEL_OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
