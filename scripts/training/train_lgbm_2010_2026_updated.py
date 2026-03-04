"""
LightGBM 最新版モデル作成スクリプト (2010-2026 Updated)

- 目的: 2025-2026年の最新データを含む形でLGBMモデルを再学習し、
        Concept Drift（環境変化への未追従）に対応する。
- アルゴリズム: LightGBM (binary classification, 3着以内=1)
- 学習期間: 2010/01/01 ~ 2026/02/28
- 出力先: models/historical_lgbm_2010_2026/  ← 旧モデルは残しロールバック可能
"""

import os
import sys
import pickle
import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer

# ============= 設定 =============
TRAIN_START = '2010-01-01'
TRAIN_END = '2026-02-28'   # 最新取得済みデータまで延長
OUTPUT_DIR = os.path.join(MODEL_DIR, "historical_lgbm_2010_2026")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ================================


def main() -> None:
    """メイン関数。"""
    print(f"=== LGBM 最新版モデル学習 ({TRAIN_START} ~ {TRAIN_END}) 開始 ===")
    print(f"  出力先: {OUTPUT_DIR}")
    print(f"  実行日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

    # 1. データ読み込み (年別pickleを統合)
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

    df_raw = results[
        (results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)
    ].copy()
    print(f"  対象期間のデータ数: {len(df_raw):,} 件 ({df_raw['date'].min()} ~ {df_raw['date'].max()})")

    # サンプルモード
    is_sample = "--sample" in sys.argv
    if is_sample:
        print("!!! サンプルモード (10%) で実行中 !!!")
        sampled_races = np.random.choice(
            df_raw['race_id'].unique(),
            size=max(1, int(df_raw['race_id'].nunique() * 0.1)),
            replace=False,
        )
        df_raw = df_raw[df_raw['race_id'].isin(sampled_races)].copy()
        print(f"  サンプリング後: {len(df_raw):,} 件")

    # 2. 特徴量エンジニアリング
    print("\n[2/5] 特徴量生成中...")
    processor = DataProcessor()
    engineer = FeatureEngineer()

    try:
        df_proc = processor.process_results(df_raw)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    except Exception:
        traceback.print_exc()
        raise RuntimeError("レース結果処理に失敗しました")

    for step_name, step_fn in [
        ("過去成績", lambda df: engineer.add_horse_history_features(df, hr)),
        ("コース適性", lambda df: engineer.add_course_suitability_features(df, hr)),
        ("騎手成績", lambda df: engineer.add_jockey_features(df)[0]),
        ("血統情報", lambda df: engineer.add_pedigree_features(df, peds)),
        ("オッズ特徴量", lambda df: engineer.add_odds_features(df)),
    ]:
        try:
            df_proc = step_fn(df_proc)
            df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
            print(f"  ✓ {step_name}")
        except Exception:
            traceback.print_exc()
            print(f"  [SKIP] {step_name}の追加に失敗")

    # 3. 学習データセット構築
    print("\n[3/5] 学習データセット構築中...")

    if 'rank' not in df_proc.columns and '着順' in df_proc.columns:
        df_proc['rank'] = df_proc['着順']
    if 'rank' not in df_proc.columns:
        raise RuntimeError("'rank' カラムが見つかりません")

    rank_series = df_proc['rank']
    if isinstance(rank_series, pd.DataFrame):
        rank_series = rank_series.iloc[:, 0]
    target_series = rank_series.apply(lambda x: 1 if x <= 3 else 0)

    date_series = df_proc['date']
    if isinstance(date_series, pd.DataFrame):
        date_series = date_series.iloc[:, 0]

    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順',
        'time', '着差', '通過', '上り', '単勝', '人気',
        'horse_name', 'jockey', 'trainer', 'owner', 'gender', 'original_race_id',
        '賞金（万円）', 'タイム指数', 'タイム秒', 'odds', 'popularity', 'is_win',
        'return', 'rank_num',
    ]

    df_proc = df_proc.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True)
    date_series = date_series.reset_index(drop=True)

    # クリーンなDataFrame構築
    df_clean = pd.DataFrame(index=df_proc.index)
    valid_features = []
    seen_cols: set = set()

    for col in df_proc.columns:
        if col in exclude_cols or col in seen_cols:
            continue
        seen_cols.add(col)
        col_data = df_proc[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        if pd.api.types.is_numeric_dtype(col_data):
            df_clean[col] = col_data
            valid_features.append(col)
        else:
            converted = pd.to_numeric(col_data, errors='coerce')
            if converted.notna().sum() > 0:
                df_clean[col] = converted
                valid_features.append(col)

    print(f"  有効な特徴量数: {len(valid_features)}")

    df_clean['target'] = target_series
    df_clean['date'] = date_series
    df_clean = df_clean.dropna(subset=['target', 'date'])

    # 時系列順ソート → 80/20 split
    df_clean = df_clean.sort_values('date')
    split_idx = int(len(df_clean) * 0.8)

    X = df_clean[valid_features]
    y = df_clean['target']
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    print(f"  学習: {len(X_train):,} 件 / 検証: {len(X_val):,} 件")

    # 4. 学習
    print("\n[4/5] LightGBM モデルを学習中...")
    model = HorseRaceModel()
    model.train(X_train, y_train, X_val=X_val, y_val=y_val)

    # 5. 保存
    print(f"\n[5/5] モデルを保存中: {OUTPUT_DIR}")
    model.save(os.path.join(OUTPUT_DIR, 'model.pkl'))
    with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(OUTPUT_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)

    meta = {
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'num_features': len(valid_features),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'created_at': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("[DONE] LGBM モデル再学習完了")
    print(f"   出力先: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
