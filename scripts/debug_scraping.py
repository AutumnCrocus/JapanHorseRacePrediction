
import os
import sys
import pandas as pd

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.scraping import Shutuba, Odds

def test_scraping(race_id):
    print(f"--- Testing Race: {race_id} ---")
    
    # Shutuba.scrape
    print("\n[Shutuba.scrape]")
    df = Shutuba.scrape(race_id)
    if df.empty:
        print("Empty DataFrame returned.")
    else:
        print(f"Retrieved {len(df)} horses.")
        # '単勝' カラムを表示
        print(df[['馬番', '馬名', '単勝']].to_string())

    # Odds.scrape
    print("\n[Odds.scrape]")
    odds_data = Odds.scrape(race_id)
    if not odds_data:
        print("No odds data returned.")
    else:
        tan_odds = odds_data.get('tan', {})
        print(f"Tan odds: {tan_odds}")

if __name__ == "__main__":
    test_scraping("202605010501") # Tokyo 01R
    test_scraping("202605010511") # Tokyo 11R
