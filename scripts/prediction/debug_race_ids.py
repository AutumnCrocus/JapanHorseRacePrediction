import os
import sys
import re
import pandas as pd
from datetime import datetime

# プロジェクトルートの設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.data_loader import fetch_and_process_race_data
from modules.scraping import Odds, Shutuba
import app

# オッズ取得をスキップ
Odds.scrape = lambda x: {}

def debug_find_race_ids():
    app.load_model('lgbm')
    app.load_model('ltr')
    app.load_model('stacking')

    # 会場リスト
    venues = {"05": "東京", "09": "阪神", "10": "小倉"}
    
    # 2026/02/21 (土)
    # 開催回(kai)と日目(day)の組み合わせをいくつか試す
    possible_kai_day = [
        ("01", "07"), # 東京
        ("01", "08"),
        ("01", "01"), # 阪神
        ("01", "02"),
        ("02", "01"), # 小倉
        ("02", "02")
    ]

    for v_id, v_name in venues.items():
        print(f"\n--- Searching for {v_name} ({v_id}) ---")
        for kai, day in possible_kai_day:
            race_id = f"2026{v_id}{kai}{day}01"
            print(f"Checking Race ID: {race_id}...", end=" ")
            try:
                df = Shutuba.scrape(race_id)
                if not df.empty:
                    print("FOUND!")
                    # 1Rが見つかれば、その組み合わせが正解
                    # そのまま12Rまで生成するロジックに使える
                else:
                    print("Empty")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    debug_find_race_ids()
