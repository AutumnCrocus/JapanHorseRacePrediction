"""
オッズ取得のテストスクリプト
"""
import sys
sys.path.insert(0, r'c:\Users\t4kic\Documents\ネット競馬')

from modules.scraping import Odds

# 小倉牝馬SのレースIDでテスト
race_id = '202610010111'

print(f"Testing odds scraping for race_id: {race_id}")
print("=" * 60)

odds_data = Odds.scrape(race_id)

print("\n=== RESULTS ===")
print(f"Tan odds count: {len(odds_data['tan'])}")
print(f"Fuku odds count: {len(odds_data['fuku'])}")

print("\n=== TAN ODDS (単勝) ===")
for umaban, odds in sorted(odds_data['tan'].items()):
    print(f"  馬番 {umaban}: {odds}倍")

print("\n=== FUKU ODDS (複勝) ===")
for umaban, odds_range in sorted(odds_data['fuku'].items()):
    print(f"  馬番 {umaban}: {odds_range[0]} - {odds_range[1]}倍")

print("\n=== SUMMARY ===")
if odds_data['tan'] or odds_data['fuku']:
    print("✓ オッズ取得成功")
else:
    print("✗ オッズ取得失敗")
