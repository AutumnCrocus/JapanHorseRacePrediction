
"""
2010-2026 データセットでのモデル学習 (最新モデル)
- 目的: 2026年以降の予測用、最新データを含むモデルを作成
- 学習期間: 2010/01/01 ~ 2026/02/08
- 特徴: 徹底的なデータクレンジングとType Safety

実行方法（Gemini Flashでも実行可能）:
    python scripts/train_2010_2026.py
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.preprocessing import DataProcessor as RaceDataProcessor, FeatureEngineer

# ============= 設定 (Gemini Flashでも修正可能) =============
TRAIN_START = '2010-01-01'
TRAIN_END = '2026-02-08'
OUTPUT_DIR = os.path.join(MODEL_DIR, "historical_2010_2026")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================================

def main():
    print(f"=== モデル学習開始 ({TRAIN_START} ~ {TRAIN_END}) ===", flush=True)
    
    # 1. データ読み込み
    print("データを読み込み中...", flush=True)
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
    
    # 2. 期間フィルタリング
    print(f"期間フィルタリング: {TRAIN_START} ~ {TRAIN_END}", flush=True)
    df_raw = results[(results['date'] >= TRAIN_START) & (results['date'] <= TRAIN_END)].copy()
    print(f"学習サンプル数: {len(df_raw):,}", flush=True)
    
    # 3. 特徴量エンジニアリング
    print("特徴量を生成中...", flush=True)
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
        except:
            traceback.print_exc()
            print("  [SKIP] 過去成績の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("  - コース適性を追加...", flush=True)
        try:
            df_proc = engineer.add_course_suitability_features(df_proc, hr)
        except:
            traceback.print_exc()
            print("  [SKIP] コース適性の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("  - 騎手成績を追加...", flush=True)
        try:
            df_proc, _ = engineer.add_jockey_features(df_proc)
        except:
            traceback.print_exc()
            print("  [SKIP] 騎手成績の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("  - 血統情報を追加...", flush=True)
        try:
            df_proc = engineer.add_pedigree_features(df_proc, peds)
        except:
            traceback.print_exc()
            print("  [SKIP] 血統情報の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
        print("  - オッズ特徴量を追加...", flush=True)
        try:
            df_proc = engineer.add_odds_features(df_proc)
        except:
            traceback.print_exc()
            print("  [SKIP] オッズ特徴量の追加に失敗", flush=True)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
        
    except Exception as e:
        print(f"[CRITICAL] 特徴量生成に失敗: {e}")
        traceback.print_exc()
        return

    # 4. 学習データセット構築
    print("学習データセットを構築中...", flush=True)
    
    # ターゲット変数の準備
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
    
    # インデックスリセット
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
            except:
                pass

    print(f"有効な特徴量数: {len(valid_features)}", flush=True)
    
    df_clean['target'] = target_series
    df_clean['date'] = date_series
    
    # 欠損値除去
    prev_size = len(df_clean)
    df_clean = df_clean.dropna(subset=['target', 'date'])
    print(f"欠損値除去: {prev_size - len(df_clean):,} 行", flush=True)
    
    # 日付順ソート
    print("日付順にソート中...", flush=True)
    try:
        df_clean = df_clean.sort_values('date')
    except Exception as e:
        print(f"[ERROR] ソート失敗: {e}")
        return
    
    # 学習/検証分割 (時系列分割: 80% train, 20% val)
    print("データ分割中...", flush=True)
    split_idx = int(len(df_clean) * 0.8)
    
    X = df_clean[valid_features]
    y = df_clean['target']
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"学習データ: {len(X_train):,}, 検証データ: {len(X_val):,}", flush=True)
    
    # 5. モデル学習
    print("LightGBMモデルを学習中...", flush=True)
    model = HorseRaceModel()
    model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # 6. 保存
    print(f"モデルを保存中: {OUTPUT_DIR}", flush=True)
    model.save(os.path.join(OUTPUT_DIR, 'model.pkl'))
    
    with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(OUTPUT_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
        
    print("="*50, flush=True)
    print("✅ モデル学習が完了しました", flush=True)
    print(f"   出力先: {OUTPUT_DIR}", flush=True)
    print("="*50, flush=True)

if __name__ == "__main__":
    main()
