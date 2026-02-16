
import os
import sys
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.scraping import Results, get_race_id_list
from modules.constants import RAW_DATA_DIR

# ============= 設定 =============
START_YEAR = 2010
END_YEAR = 2026
BASE_OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "results")
MAX_WORKERS = 4 
# ===============================

def process_year(year):
    print(f"\nProcessing Year: {year}")
    
    # 保存ディレクトリ作成
    year_dir = os.path.join(BASE_OUTPUT_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)
    output_file = os.path.join(year_dir, f"results_{year}.pkl")
    
    # 既存データの読み込み
    existing_data = pd.DataFrame()
    executed_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            if not existing_data.empty:
                # indexがrace_idになっている前提
                if isinstance(existing_data.index, pd.MultiIndex):
                    executed_ids = set(existing_data.index.get_level_values(0).unique().astype(str))
                else:
                    executed_ids = set(existing_data.index.unique().astype(str))
                print(f"  Loaded {len(executed_ids)} existing races.")
        except Exception as e:
            print(f"  Failed to load existing data: {e}. Starting fresh.")
    
    # IDリスト生成
    all_ids = get_race_id_list(year, year)
    
    # 未取得IDのみ対象にする
    to_scrape = [rid for rid in all_ids if rid not in executed_ids]
    
    print(f"  Target: {len(to_scrape)} new races (Total: {len(all_ids)})")
    
    if not to_scrape:
        print("  No new races to scrape.")
        return

    new_results = []
    
    # Chunking for results scraping as it returns DataFrame
    chunk_size = 50
    with tqdm(total=len(to_scrape), desc=f"Scraping {year}") as pbar:
        for i in range(0, len(to_scrape), chunk_size):
            chunk = to_scrape[i:i+chunk_size]
            try:
                # Use existing scraping module
                df = Results.scrape(chunk) 
                if not df.empty:
                    new_results.append(df)
            except Exception as e:
                print(f"Error scraping chunk: {e}")
            
            pbar.update(len(chunk))
    
    if new_results:
        print(f"  concatenating {len(new_results)} chunks...")
        new_df = pd.concat(new_results)
        
        if existing_data.empty:
            final_df = new_df
        else:
            final_df = pd.concat([existing_data, new_df])
            # 重複削除 (念のため: index=race_id で重複排除したいが、同着等のMultiIndex構造破損に注意)
            # Resultsは通常 MultiIndex['race_id', 'horse_id'] なので、単純な drop_duplicates は危険
            # ここでは単純結合のみとし、分析時に重複排除を任せる（安全策）
        
        with open(output_file, 'wb') as f:
            pickle.dump(final_df, f)
            
        print(f"  Saved to {output_file} (Total Rows: {len(final_df)})")
    else:
        print("  No data scraped.")

def main():
    parser = argparse.ArgumentParser(description="Scrape historical Results data.")
    parser.add_argument("--start", type=int, default=START_YEAR, help="Start year")
    parser.add_argument("--end", type=int, default=END_YEAR, help="End year")
    args = parser.parse_args()
    
    print("=== Historical Results Scraping Start ===")
    print(f"Years: {args.start} - {args.end}")
    
    for year in range(args.start, args.end + 1):
        process_year(year)
        
    print("=== All Completed ===")

if __name__ == "__main__":
    main()
