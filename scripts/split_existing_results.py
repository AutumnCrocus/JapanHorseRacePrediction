
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import RAW_DATA_DIR, RESULTS_FILE

def split_results_by_year():
    """
    既存の results.pickle を読み込み、年ごとに分割して保存する。
    保存先: data/raw/results/{year}/results_{year}.pkl
    """
    print("Loading monolithic results.pickle...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    
    if not os.path.exists(results_path):
        print("Error: results.pickle not found.")
        return

    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    if results.empty:
        print("Results dataframe is empty.")
        return

    print(f"Total rows: {len(results)}")
    
    # 日付カラムの確認と年抽出
    if 'date' not in results.columns:
        # dateがない場合、race_id (index) から抽出を試みる
        # YYYY...
        try:
            results['temp_date'] = pd.to_datetime(results.index.astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except:
            print("Error: Could not extract date from index.")
            return
    else:
        results['temp_date'] = pd.to_datetime(results['date'], errors='coerce')

    # 年カラム作成
    results['year'] = results['temp_date'].dt.year
    
    # 年ごとにグループ化して保存
    years = sorted(results['year'].dropna().unique().astype(int))
    print(f"Years found: {years}")

    base_output_dir = os.path.join(RAW_DATA_DIR, "results")
    os.makedirs(base_output_dir, exist_ok=True)

    for year in tqdm(years, desc="Splitting by year"):
        year_df = results[results['year'] == year].copy()
        
        # 一時カラム削除
        if 'temp_date' in year_df.columns:
            year_df = year_df.drop(columns=['temp_date'])
        if 'year' in year_df.columns:
            year_df = year_df.drop(columns=['year']) # 元のデータ構造を維持するため削除

        # 保存ディレクトリ
        year_dir = os.path.join(base_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        output_file = os.path.join(year_dir, f"results_{year}.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(year_df, f)
            
    print("Splitting completed.")

if __name__ == "__main__":
    split_results_by_year()
