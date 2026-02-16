"""
2025年全レース予測スクリプト (LTR vs LGBM)
- 目的: 2025年の全レースに対して、ヒストリカルLTRモデル (2010-2024学習) と
        既存のヒストリカルLGBMモデル (Box4戦略用) で予測を行い、比較用CSVを出力する。
- 出力:
  - data/processed/prediction_2025_ltr.csv
  - data/processed/prediction_2025_lgbm.csv
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import (  # noqa: E402
    MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE,
    HORSE_RESULTS_FILE, PEDS_FILE
)

# ============= 設定 =============
TARGET_YEAR = 2025
LTR_MODEL_DIR = os.path.join(MODEL_DIR, "historical_ltr_2010_2024")
LGBM_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
# ===============================


class RankingWrapper:
    """LTRモデルのラッパー"""

    def __init__(self, data: dict):
        self.model = data['model']
        self.feature_names = data['feature_names']

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])


def load_models() -> dict:
    """モデルとプロセッサをロード"""
    models: dict = {}

    # 1. LTR Model
    print("Loading LTR model...")
    with open(os.path.join(LTR_MODEL_DIR, 'ranking_model.pkl'), 'rb') as f:
        data = pickle.load(f)
    models['ltr'] = {
        'model': RankingWrapper(data),
        'processor': pickle.load(
            open(os.path.join(LTR_MODEL_DIR, 'processor.pkl'), 'rb')
        ),
        'engineer': pickle.load(
            open(os.path.join(LTR_MODEL_DIR, 'engineer.pkl'), 'rb')
        ),
    }

    # 2. LGBM Model (dict型: keys=model, feature_names, ...)
    print("Loading LGBM model...")
    lgbm_model_path = os.path.join(LGBM_MODEL_DIR, "model.pkl")
    if os.path.exists(lgbm_model_path):
        with open(lgbm_model_path, 'rb') as f:
            lgbm_data = pickle.load(f)
        # dict型の場合: model, feature_names キーを持つ
        if isinstance(lgbm_data, dict):
            booster = lgbm_data['model']
            feat_names = lgbm_data.get('feature_names', [])
        else:
            booster = lgbm_data
            feat_names = booster.feature_name()
        models['lgbm'] = {
            'booster': booster,
            'feature_names': feat_names,
            'processor': pickle.load(
                open(os.path.join(LGBM_MODEL_DIR, 'processor.pkl'), 'rb')
            ),
            'engineer': pickle.load(
                open(os.path.join(LGBM_MODEL_DIR, 'engineer.pkl'), 'rb')
            ),
        }
    else:
        print("Warning: No LGBM models found. Skipping LGBM.")
        models['lgbm'] = None

    return models


def find_col(df: pd.DataFrame, candidates: list) -> str:
    """DataFrameから候補カラム名リストの中で最初に見つかるものを返す"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_result_df(
    df_proc: pd.DataFrame,
    scores: np.ndarray,
    score_col_name: str
) -> pd.DataFrame:
    """前処理済みデータからCSV出力用DataFrameを構築する

    process_results後のデータ(df_proc)をベースにする。
    これにより行数不一致の問題を回避できる。
    """
    df_res = pd.DataFrame(index=df_proc.index)
    df_res['race_id'] = df_proc['race_id']

    # 馬番
    umaban_col = find_col(df_proc, ['horse_number', 'umaban', '\u99ac\u756a', '\u99ac \u756a'])
    if umaban_col:
        df_res['horse_number'] = df_proc[umaban_col]
    else:
        df_res['horse_number'] = 0

    # 馬名
    umamei_col = find_col(df_proc, ['horse_name', '\u99ac\u540d'])
    if umamei_col:
        df_res['horse_name'] = df_proc[umamei_col]
    else:
        df_res['horse_name'] = ''

    # 日付
    date_col = find_col(df_proc, ['date'])
    if date_col:
        df_res['date'] = df_proc[date_col]

    # オッズ
    odds_col = find_col(df_proc, [
        'odds', 'win_odds', '\u5358\u52dd', '\u5358 \u52dd', 'tansho'
    ])
    if odds_col:
        df_res['win_odds'] = pd.to_numeric(
            df_proc[odds_col], errors='coerce'
        ).fillna(0)
    else:
        df_res['win_odds'] = 0

    # 着順
    rank_col = find_col(df_proc, [
        'rank', '\u7740\u9806', '\u7740 \u9806', 'rank_num'
    ])
    if rank_col:
        df_res['actual_rank'] = pd.to_numeric(
            df_proc[rank_col], errors='coerce'
        ).fillna(99)

    # スコア
    df_res[score_col_name] = scores

    # ランク
    df_res['rank_prediction'] = df_res.groupby('race_id')[score_col_name].rank(
        method='first', ascending=False
    )

    return df_res


def main():  # noqa: C901
    """メイン処理"""
    print("=== 2025年全レース予測開始 ===")

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

    results = results.loc[:, ~results.columns.duplicated()]

    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(
            results['race_id'].astype(str).str[:8],
            format='%Y%m%d', errors='coerce'
        )
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    df_2025 = results[results['date'].dt.year == TARGET_YEAR].copy()
    n_races = df_2025['race_id'].nunique()
    print(f"2025年の対象データ: {len(df_2025)}件 ({n_races}レース)")

    if df_2025.empty:
        print("Error: 2025年のデータが見つかりません。")
        return

    # モデルロード
    models = load_models()

    # === LTR予測 ===
    if models.get('ltr'):
        print("\n=== LTR Predictions ===")
        env = models['ltr']

        print("Generating features for LTR...")
        df_proc = env['processor'].process_results(df_2025.copy())
        df_proc = env['engineer'].add_horse_history_features(df_proc, hr)
        df_proc = env['engineer'].add_course_suitability_features(df_proc, hr)
        df_proc, _ = env['engineer'].add_jockey_features(df_proc)
        df_proc = env['engineer'].add_pedigree_features(df_proc, peds)
        df_proc = env['engineer'].add_odds_features(df_proc)

        feature_names = env['model'].feature_names
        for col in feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0
        X = df_proc[feature_names].fillna(0)

        print(f"Predicting LTR scores for {len(X)} rows...")
        scores = env['model'].predict(X)

        df_res = build_result_df(df_proc, scores, 'ltr_score')

        output_path = "data/processed/prediction_2025_ltr.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_res.to_csv(output_path, index=False)
        print(f"LTR predictions saved to: {output_path} ({len(df_res)} rows)")

    # === LGBM予測 (比較用) ===
    if models.get('lgbm'):
        print("\n=== LGBM Predictions ===")
        env = models['lgbm']
        booster = env['booster']
        feature_names = env['feature_names']

        print("Generating features for LGBM...")
        df_proc = env['processor'].process_results(df_2025.copy())
        df_proc = env['engineer'].add_horse_history_features(df_proc, hr)
        df_proc = env['engineer'].add_course_suitability_features(df_proc, hr)
        df_proc, _ = env['engineer'].add_jockey_features(df_proc)
        df_proc = env['engineer'].add_pedigree_features(df_proc, peds)
        df_proc = env['engineer'].add_odds_features(df_proc)

        # LGBMモデルの場合、category型への変換は慎重に行う必要がある
        # 学習時のpandas_categoricalを確認できない/していない場合、
        # 安全のためにcategory型変換を行わず、LabelEncodingのみで渡すか
        # そもそも数値化されていることを期待する
        
        # Processorのencode_categoricalを使ってみる
        cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
        cat_cols = [c for c in cat_cols if c in df_proc.columns]
        df_proc = env['processor'].encode_categorical(df_proc, cat_cols)
        
        for col in feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0
        
        X = df_proc[feature_names].fillna(0)
        
        # object型が残っている場合はcategoryに変換せず、強制的にcategoryコード化または0埋め
        # (LGBMエラー回避のため)
        for col in X.columns:
            if X[col].dtype == 'object':
                 X[col] = X[col].astype('category')

        print(f"Predicting LGBM probabilities for {len(X)} rows...")
        probs = booster.predict(X)

        df_res = build_result_df(df_proc, probs, 'prob_score')

        output_path = "data/processed/prediction_2025_lgbm.csv"
        df_res.to_csv(output_path, index=False)
        print(f"LGBM predictions saved to: {output_path} ({len(df_res)} rows)")

    print("\n=== All Done ===")


if __name__ == "__main__":
    main()
