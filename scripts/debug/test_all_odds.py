"""
全券種のオッズ取得をテスト
"""
import sys
sys.path.insert(0, r'c:\Users\t4kic\Documents\ネット競馬')

from modules.scraping import Odds

race_id = '202610010111'
print(f"Testing all bet types for race_id: {race_id}")
print("=" * 60)

odds_data = Odds.scrape(race_id)

print("\n=== RESULTS ===")
for bet_type, data in odds_data.items():
    print(f"{bet_type.upper()}: {len(data)} items")
    if data:
        # Show first 3 examples
        items = list(data.items())[:3]
        for key, value in items:
            print(f"  {key}: {value}")
