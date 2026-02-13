
import os
import sys
import pandas as pd

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.scraping import Shutuba, Odds

def test_race(race_id):
    print(f"Testing Race: {race_id}")
    
    print("\n--- Testing Shutuba.scrape ---")
    df = Shutuba.scrape(race_id)
    if not df.empty:
        print(df[['馬番', '馬名', '単勝', '人気']].head(10))
    else:
        print("Shutuba.scrape returned empty DataFrame")

    print("\n--- Testing Odds.scrape ---")
    odds = Odds.scrape(race_id)
    print("TAN Odds count:", len(odds.get('tan', {})))
    print("TAN Odds (first 5):", dict(list(odds.get('tan', {}).items())[:5]))

if __name__ == "__main__":
    # 東京01R (2026-02-14)
    test_race("202605010501")
    # 京都01R (2026-02-14)
    # test_race("202608020501")
