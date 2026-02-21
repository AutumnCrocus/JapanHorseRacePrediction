"""
CatBoost モデル学習 (2010-2024年データ)
- モデル: CatBoostClassifier
- 学習期間: 2010/01/01 ~ 2024/12/31
- 特徴: カテゴリ変数をネイティブ処理、比較用として LightGBM と同じ特徴量を使用
- 出力先: models/catboost_2010_2024/model.pkl
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor as RaceDataProcessor, FeatureEngineer

try:
    import catboost as cb
    print("CatBoostのインポート成功", flush=True)
except ImportError:
    print("[ERROR] catboostがインストールされていません。pip install catboost を実行してください。")
    sys.exit(1)

# ============= 設定 =============
TRAIN_START = '2010-01-01'
TRAIN_END = '2024-12-31'
OUTPUT_DIR = os.path.join(MODEL_DIR, "catboost_2010_2024")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"=== CatBoost モデル学習開始 ({TRAIN_START} ~ {TRAIN_END}) ===", flush=True)

    # 1. データ読み込み
    print("[1] データを読み込み中...", flush=True)
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

    # 2. 期間フィルタリング
    print(f"[2] 期間フィルタリング: {TRAIN_START} ~ {TRAIN_END}", flush=True)
    df_raw = results[
        (results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)
    ].copy()
    print(f"   学習サンプル数: {len(df_raw):,}", flush=True)

    # 3. 特徴量エンジニアリング
    print("[3] 特徴量を生成中...", flush=True)
    processor = RaceDataProcessor()
    engineer = FeatureEngineer()

    import traceback

    try:
        print("  - レース結果処理...", flush=True)
        df_proc = processor.process_results(df_raw)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

        print("  - 過去成績を追加...", flush=True)
        try:
            df_proc = engineer.add_horse_history_features(df_proc, hr)
        except Exception:
            traceback.print_exc()
            print("  [SKIP] 過去成績の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

        print("  - コース適性を追加...", flush=True)
        try:
            df_proc = engineer.add_course_suitability_features(df_proc, hr)
        except Exception:
            traceback.print_exc()
            print("  [SKIP] コース適性の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

        print("  - 騎手成績を追加...", flush=True)
        try:
            df_proc, _ = engineer.add_jockey_features(df_proc)
        except Exception:
            traceback.print_exc()
            print("  [SKIP] 騎手成績の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

        print("  - 血統情報を追加...", flush=True)
        try:
            df_proc = engineer.add_pedigree_features(df_proc, peds)
        except Exception:
            traceback.print_exc()
            print("  [SKIP] 血統情報の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

        print("  - オッズ特徴量を追加...", flush=True)
        try:
            df_proc = engineer.add_odds_features(df_proc)
        except Exception:
            traceback.print_exc()
            print("  [SKIP] オッズ特徴量の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

    except Exception as e:
        print(f"[CRITICAL] 特徴量生成に失敗: {e}")
        traceback.print_exc()
        return

    # 4. 学習データセット構築
    print("[4] 学習データセットを構築中...", flush=True)

    if 'rank' not in df_proc.columns and '着順' in df_proc.columns:
        df_proc['rank'] = df_proc['着順']

    if 'rank' not in df_proc.columns:
        print("[ERROR] 'rank' カラムが見つかりません")
        return

    rank_series = df_proc['rank']
    if isinstance(rank_series, pd.DataFrame):
        rank_series = rank_series.iloc[:, 0]

    target_series = rank_series.apply(lambda x: 1 if x <= 3 else 0)
    date_series = df_proc['date']
    if isinstance(date_series, pd.DataFrame):
        date_series = date_series.iloc[:, 0]

    # 除外カラム（レース後に判明する情報）
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順',
        'time', '着差', '通過', '上り', '単勝', '人気',
        'horse_name', 'jockey', 'trainer', 'owner', 'gender', 'original_race_id',
        '賞金（万円）', 'タイム指数', 'タイム秒', 'odds', 'popularity', 'is_win',
        'return', 'rank_num'
    ]

    df_proc = df_proc.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True)
    date_series = date_series.reset_index(drop=True)

    # クリーンなDataFrame構築
    df_clean = pd.DataFrame(index=df_proc.index)
    valid_features = []
    seen_cols = set()

    for col in df_proc.columns:
        if col in exclude_cols:
            continue
        if col in seen_cols:
            continue
        seen_cols.add(col)

        col_data = df_proc[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        if pd.api.types.is_numeric_dtype(col_data):
            df_clean[col] = col_data
            valid_features.append(col)
        else:
            try:
                converted = pd.to_numeric(col_data, errors='coerce')
                if converted.notna().sum() > 0:
                    df_clean[col] = converted
                    valid_features.append(col)
            except Exception:
                pass

    print(f"   有効な特徴量数: {len(valid_features)}", flush=True)

    df_clean['target'] = target_series
    df_clean['date'] = date_series

    prev_size = len(df_clean)
    df_clean = df_clean.dropna(subset=['target', 'date'])
    print(f"   欠損値除去: {prev_size - len(df_clean):,} 行", flush=True)

    # 日付順ソート・時系列分割
    df_clean = df_clean.sort_values('date')
    split_idx = int(len(df_clean) * 0.8)

    X = df_clean[valid_features]
    y = df_clean['target']

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    print(f"   学習データ: {len(X_train):,}, 検証データ: {len(X_val):,}", flush=True)

    # NaN を 0 で補完（CatBoostはNaNを許容するが念のため）
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # 5. CatBoost 学習
    print("[5] CatBoostClassifier を学習中...", flush=True)
    catboost_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
    }

    cat_model = cb.CatBoostClassifier(**catboost_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # 6. 検証精度の表示
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred_proba = cat_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n検証AUC: {auc:.4f}, Accuracy: {acc:.4f}", flush=True)

    # 7. モデルを HorseRaceModel 互換形式で保存
    print(f"[6] モデルを保存中: {OUTPUT_DIR}", flush=True)

    # HorseRaceModel 互換にするためラップして pickle 保存
    model_data = {
        'model': cat_model,
        'model_type': 'catboost',
        'feature_names': valid_features,
        'feature_importance': pd.DataFrame({
            'feature': valid_features,
            'importance': cat_model.get_feature_importance()
        }).sort_values('importance', ascending=False),
        'model_params': catboost_params
    }
    model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # 特徴量重要度もCSV保存
    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    model_data['feature_importance'].to_csv(fi_path, index=False, encoding='utf-8-sig')
    print(f"   特徴量重要度: {fi_path}", flush=True)

    # processor/engineer も保存
    with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(OUTPUT_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)

    # 検証AUCを記録
    result_summary = {
        'model_type': 'catboost',
        'train_period': f'{TRAIN_START} ~ {TRAIN_END}',
        'n_features': len(valid_features),
        'val_auc': float(auc),
        'val_accuracy': float(acc),
    }
    import json
    with open(os.path.join(OUTPUT_DIR, 'train_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print("=" * 50, flush=True)
    print("✅ CatBoost モデル学習が完了しました", flush=True)
    print(f"   出力先: {OUTPUT_DIR}", flush=True)
    print(f"   検証AUC: {auc:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
