import os
import sys
import pickle
import pandas as pd
import json
import time
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import (
    RAW_DATA_DIR, RESULTS_FILE, MAX_WORKERS
)
from modules.scraping import Results, get_race_id_list, update_data

# 設定
START_YEAR = 2016
END_YEAR = 2026
BACKUP_DIR = os.path.join(RAW_DATA_DIR, "recovery_backups")
PROGRESS_FILE = os.path.join(RAW_DATA_DIR, "recovery_progress.json")
TEMP_FILE_PATTERN = os.path.join(BACKUP_DIR, "results_{year}.pickle")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_years": [], "current_year": START_YEAR, "last_updated": ""}

def save_progress(progress):
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def recover_data():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    progress = load_progress()
    
    print(f"=== 全データ復旧プロセス開始 ({START_YEAR}-{END_YEAR}) ===")
    print(f"進捗: 完了済み年 {progress['completed_years']}")
    
    for year in range(START_YEAR, END_YEAR + 1):
        if year in progress["completed_years"]:
            print(f"Skip {year}: Already completed.")
            continue
            
        print(f"\n--- {year}年のデータを処理中 ---")
        temp_path = TEMP_FILE_PATTERN.format(year=year)
        
        # レースIDリストの生成
        race_ids = get_race_id_list(year, year)
        print(f"{year}年の総レース数: {len(race_ids)}")
        
        # 既存の部分データがあれば読み込み（中断再開用）
        year_data = pd.DataFrame()
        if os.path.exists(temp_path):
            with open(temp_path, 'rb') as f:
                year_data = pickle.load(f)
            print(f"既存の tạm データをロードしました: {len(year_data)}行 (約 {len(year_data.index.unique())} レース分)")
            
            # 既に取得済みのレースを除外
            processed_ids = set(year_data.index.unique().astype(str))
            race_ids = [rid for rid in race_ids if str(rid) not in processed_ids]
            print(f"残り取得対象レース数: {len(race_ids)}")

        if len(race_ids) > 0:
            # スクレイピング実行
            chunk_size = 100 # こまめに保存
            for i in range(0, len(race_ids), chunk_size):
                chunk = race_ids[i:i + chunk_size]
                print(f"取得中... {i}/{len(race_ids)} ({year}年)")
                
                chunk_results = Results.scrape(chunk)
                
                if not chunk_results.empty:
                    year_data = pd.concat([year_data, chunk_results])
                    # 保存 (Atomic-ish)
                    temp_out = temp_path + ".tmp"
                    with open(temp_out, 'wb') as f:
                        pickle.dump(year_data, f)
                    if os.path.exists(temp_path): os.remove(temp_path)
                    os.rename(temp_out, temp_path)
                
                time.sleep(1) # 少し休憩

        print(f"{year}年の取得完了。合計行数: {len(year_data)}")
        progress["completed_years"].append(year)
        save_progress(progress)

    # 最終統合
    print("\n=== 全データの統合中... ===")
    all_results = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = TEMP_FILE_PATTERN.format(year=year)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                all_results.append(pickle.load(f))
    
    if all_results:
        final_df = pd.concat(all_results)
        # race_id + horse_id で一意にする（Results.scrape の index は race_id）
        # horse_id があるのでそれも考慮して重複排除
        # 本来 Results.scrape で重複は入りにくいはずだが念のため
        # results.pickle は race_id が index 
        
        target_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
        
        # 安全のため既存ファイルをバックアップ
        if os.path.exists(target_path):
            bak_path = target_path + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(target_path, bak_path)
            print(f"既存のファイルをバックアップしました: {bak_path}")

        with open(target_path, 'wb') as f:
            pickle.dump(final_df, f)
        
        print(f"全統合完了! ファイル保存先: {target_path}")
        print(f"最終行数: {len(final_df)}")
        print(f"最終レース数: {len(final_df.index.unique())}")
    else:
        print("データが見つかりませんでした。")

if __name__ == "__main__":
    recover_data()
