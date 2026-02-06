import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE
from modules.training import HorseRaceModel, RacePredictor

def debug_leakage():
    print("=== データリーク調査用デバッグ ===")
    
    # データロード
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    
    print("\nResults Columns:", results.columns if hasattr(results, 'columns') else "No Columns")
    print("Results Index Names:", results.index.names)
    print("HR Columns:", hr.columns if hasattr(hr, 'columns') else "No Columns")
    print("HR Index Names:", hr.index.names)

    # 2025年の最初のレースを抽出
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        # カラム名の特定
        race_id_col = 'race_id' if 'race_id' in results.columns else 'level_0'
    else:
        results = results.reset_index()
        race_id_col = 'race_id' if 'race_id' in results.columns else 'index'
    
    results['date'] = pd.to_datetime(results[race_id_col].astype(str).str[:8], format='%Y%m%d', errors='coerce').dt.normalize()
    df_2025 = results[results[race_id_col].astype(str).str.startswith('2025')].sort_values(race_id_col).head(1)
    
    if df_2025.empty:
        print("2025年のレースが見つかりません。")
        return

    race_id = df_2025[race_id_col].iloc[0]
    horse_id = df_2025['horse_id'].iloc[0] if 'horse_id' in df_2025.columns else df_2025.index[0] # Fallback
    race_date = df_2025['date'].iloc[0]
    
    print(f"Target Race ID: {race_id}")
    print(f"Target Horse ID: {horse_id}")
    print(f"Race Date: {race_date}")
    
    # 馬の全履歴を確認
    print("\nHorse Full History in hr_pickle:")
    horse_full_hr = hr[hr.index == horse_id].copy()
    print(f"Total history rows for this horse: {len(horse_full_hr)}")
    if not horse_full_hr.empty:
        # 日付抽出
        horse_full_hr['date_str'] = horse_full_hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
        horse_full_hr['date'] = pd.to_datetime('20' + horse_full_hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
        print(horse_full_hr[['レース名', 'date', '着順']].sort_values('date', ascending=False).head(10))

    # モデル/リソースの準備
    from modules.preprocessing import DataProcessor, FeatureEngineer
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 特徴量生成テスト
    df_proc = processor.process_results(df_2025)
    
    # 確実に race_id を追加 (process_results 内で消えている可能性を考慮)
    if 'race_id' not in df_proc.columns:
        df_proc['race_id'] = race_id
    
    # 内部での is_predict_mode 判定の検証
    # hrの日付も正規化
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    
    # 手動診断
    match = hr[(hr.index == horse_id) & (hr['date'] == race_date)]
    if not match.empty:
        print(f"SUCCESS: 2025年のレース結果が horse_results に含まれています。 (Row Count: {len(match)})")
    else:
        print(f"WARNING: 2025年のレース結果が horse_results に含まれていません。これでは is_predict_mode=True になりリークします。")
        # 2025年のデータ全体で確認
        hr_2025 = hr[hr['date'].dt.year == 2025]
        print(f"2025年成績データの総数: {len(hr_2025)}")
        if len(hr_2025) > 0:
            print("サンプル 2025年成績データ:")
            print(hr_2025[['レース名', 'date']].head())

    # engineer.add_horse_history_features の実行状況を確認
    # (ここでは print デバッグを仕込めないので、返り値をチェック)
    print("\nExecuting add_horse_history_features...")
    df_final = engineer.add_horse_history_features(df_proc, hr)
    
    # もしリーク（is_predict_mode=True）なら、avg_rank 等が 2025年の結果を含んでしまう
    print("\n--- 特徴量生成結果 ---")
    cols = ['avg_rank', 'win_rate', 'place_rate', 'race_count', 'prev_rank']
    for c in cols:
        if c in df_final.columns:
            print(f"{c}: {df_final[c].iloc[0]}")
    
    # 真の「過去のみ」の成績を計算してみる
    past_hr = hr[(hr.index == horse_id) & (hr['date'] < race_date)]
    if not past_hr.empty:
        true_avg_rank = pd.to_numeric(past_hr['着順'], errors='coerce').mean()
        print(f"True Past Avg Rank: {true_avg_rank}")
        if abs(df_final['avg_rank'].iloc[0] - true_avg_rank) < 0.001:
            print("RESULT: No Leakage (Correctly used Expanding Window/Shift)")
        else:
            print(f"RESULT: LEAKAGE DETECTED! (Feature: {df_final['avg_rank'].iloc[0]} vs True Past: {true_avg_rank})")
    else:
        print("No past horse results found for comparison.")

if __name__ == "__main__":
    debug_leakage()
