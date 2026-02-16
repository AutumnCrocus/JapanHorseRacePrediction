"""
2026年1月のレース結果と払戻金をスクレイピングするスクリプト
"""
import os
import sys
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm
import requests

# パスの解決
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.scraping import Results, Return, get_race_id_list
from modules.constants import DATA_DIR, RAW_DATA_DIR

# 保存先
RESULTS_2026_FILE = os.path.join(RAW_DATA_DIR, 'results_202601.pickle')
RETURN_2026_FILE = os.path.join(RAW_DATA_DIR, 'return_202601.pickle')

def scrape_jan_2026():
    print("=== Scraping January 2026 Data ===")
    
    # 1. 2026年のレースID候補を生成
    # JRAは週末開催。1月はだいたい5日, 10-12日, 17-18日, 24-25日, 31日など。
    # 全会場(01-10)の第1回開催を中心に探索
    
    # get_race_id_listは全IDを生成する。
    # 1月分だけ効率的に取るため、scrapeメソッドのlimitは使わず、
    # 日付フィルタはスクレイピング後に行う（Resultsクラスが日付を取ってくるまでわからないため）
    # しかし全探索は遅い。
    # 経験則: 1月は「第1回」開催が多い。
    # 2026年1月: 
    # 中山(06), 京都(08) がメイン? あるいは小倉(10)?
    # 1月開催: 中山(06), 京都(08), 中京(07) or 小倉(10)
    # 2026年のカレンダー: 1/5(月), 1/10-12, 1/17-18, 1/24-25, 1/31
    # 場所コード: 06(中山), 08(京都), 10(小倉) が一般的。
    
    # 効率化のため、target placesを指定
    target_places = ['06', '08', '07', '10', '05', '09'] # 主要どころ + α
    
    # IDリスト生成
    print("Generating Race IDs...")
    full_id_list = get_race_id_list(2026, 2026, place_codes=target_places)
    print(f"Generated {len(full_id_list)} candidate IDs.")
    
    # スクレイピング
    print("Scraping Results...")
    results_df = Results.scrape(full_id_list)
    
    if results_df.empty:
        print("No results found for 2026 candidate IDs.")
        return

    # 日付フィルタ (1月のみ)
    if 'date' in results_df.columns:
        results_df['date'] = pd.to_datetime(results_df['date'], format='%Y年%m月%d日', errors='coerce')
        mask = (results_df['date'] >= '2026-01-01') & (results_df['date'] <= '2026-01-31')
        results_jan = results_df[mask].copy()
    else:
        results_jan = results_df # Fallback
        
    print(f"Scraped {len(results_jan)} rows for Jan 2026.")
    
    # 保存
    with open(RESULTS_2026_FILE, 'wb') as f:
        pickle.dump(results_jan, f)
    print(f"Saved results to {RESULTS_2026_FILE}")
    
    # 払戻金
    print("Scraping Returns...")
    # 取得できたレースIDのリスト
    scraped_race_ids = results_jan.index.unique().tolist()
    return_df = Return.scrape(scraped_race_ids)
    
    with open(RETURN_2026_FILE, 'wb') as f:
        pickle.dump(return_df, f)
    print(f"Saved returns to {RETURN_2026_FILE}")

if __name__ == "__main__":
    scrape_jan_2026()
