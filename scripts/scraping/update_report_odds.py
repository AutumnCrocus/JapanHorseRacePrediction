"""
レポートオッズ更新スクリプト
既存のprediction_report_*.mdファイルの単勝オッズと三連複BOX期待値を最新データで更新

使用方法:
    python scripts/update_report_odds.py prediction_report_20260131.md
"""

import os
import sys
import re
import time
from itertools import combinations
from typing import Dict, List, Tuple, Optional

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.scraping import Odds


class ReportOddsUpdater:
    """レポートファイルのオッズ情報を更新するクラス"""
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.report_content = ""
        self.race_data = {}  # {race_id: {horses: [...], box_horses: [...]}}
        
    def load_report(self):
        """レポートファイルを読み込む"""
        print(f"Loading report: {self.report_path}")
        with open(self.report_path, 'r', encoding='utf-8') as f:
            self.report_content = f.read()
        print("Report loaded successfully.")
        
    def parse_report(self):
        """レポートを解析してレースID、馬番、BOX情報を抽出"""
        print("\nParsing report...")
        
        # レースセクションを分割（### で始まる各レース）
        race_sections = re.split(r'\n### ', self.report_content)
        
        for section in race_sections[1:]:  # 最初はヘッダーなのでスキップ
            # レース名とIDを抽出（例: 東京01R: 3歳未勝利）
            race_header_match = re.search(r'^(\w+)(\d{2})R:', section)
            if not race_header_match:
                continue
                
            venue = race_header_match.group(1)
            race_num = race_header_match.group(2)
            
            # 発走情報からレースIDを推定
            # race_infoパターン: - 発走情報: 10:05発走 / ダ1400m (左) / 天候:晴 / 馬場:良
            
            # BOX推奨買い目を抽出（例: - **3連複 BOX**: 4-3-6-9-5 (1000円)）
            box_match = re.search(r'\*\*3連複 BOX\*\*:\s*([\d\-]+)', section)
            box_horses = []
            if box_match:
                box_str = box_match.group(1)
                box_horses = [int(h) for h in box_str.split('-')]
            
            # AI注目馬テーブルから馬番を抽出
            table_rows = re.findall(r'\|\s*(\d+)\s*\|[^|]+\|[^|]+\|[^|]+\|', section)
            horse_numbers = [int(num) for num in table_rows]
            
            # レースIDの推定が必要だが、ここではvenueとrace_numから生成
            # 実際のrace_idはYYYYPPKKDDRRの形式なので、日付から推定する必要がある
            # レポートのヘッダーから日付を取得
            
            self.race_data[f"{venue}{race_num}"] = {
                'venue': venue,
                'race_num': race_num,
                'box_horses': box_horses,
                'horse_numbers': horse_numbers,
                'section': section
            }
        
        print(f"Parsed {len(self.race_data)} races.")
        
    def get_race_id_from_report_date(self) -> str:
        """レポートの日付からYYYYMMDD形式を取得"""
        # ヘッダーから日付を抽出（例: # 競馬予想レポート (2026/01/31)）
        date_match = re.search(r'競馬予想レポート \((\d{4})/(\d{2})/(\d{2})\)', self.report_content)
        if date_match:
            year, month, day = date_match.groups()
            return f"{year}{month}{day}"
        return "20260131"  # デフォルト
    
    def build_race_ids(self) -> Dict[str, str]:
        """会場とレース番号からレースIDを構築"""
        date_str = self.get_race_id_from_report_date()
        year = date_str[:4]
        
        # 会場コードマッピング（2026年1月31日の実際の開催）
        venue_codes = {
            '東京': '05',  # 東京
            '京都': '08',  # 京都
            '小倉': '10'   # 小倉
        }
        
        # 開催回・日の特定（2026年1月31日の実際の値）
        kai_day_map = {
            '東京': '0101',  # 1回1日目
            '京都': '0201',  # 2回1日目
            '小倉': '0103'   # 1回3日目
        }
        
        race_id_map = {}
        for key, data in self.race_data.items():
            venue = data['venue']
            race_num = data['race_num']
            
            if venue in venue_codes:
                place_code = venue_codes[venue]
                kai_day = kai_day_map.get(venue, '0101')
                race_id = f"{year}{place_code}{kai_day}{race_num}"
                race_id_map[key] = race_id
        
        return race_id_map
    
    def fetch_latest_odds(self, race_id: str) -> Optional[Dict]:
        """最新のオッズを取得"""
        print(f"  Fetching odds for race {race_id}...")
        try:
            odds_data = Odds.scrape(race_id)
            if odds_data and 'tan' in odds_data and odds_data['tan']:
                print(f"    [OK] Fetched {len(odds_data['tan'])} win odds")
                if 'sanrenpuku' in odds_data and odds_data['sanrenpuku']:
                    print(f"    [OK] Fetched {len(odds_data['sanrenpuku'])} trifecta odds")
                return odds_data
            else:
                print(f"    [X] No odds data available")
                return None
        except Exception as e:
            print(f"    [X] Error fetching odds: {e}")
            return None
    
    def calculate_box_expected_value(self, box_horses: List[int], sanrenpuku_odds: Dict) -> float:
        """三連複BOXの期待値を計算（全組み合わせの平均オッズ）"""
        if len(box_horses) < 3:
            return 0.0
        
        # C(n, 3)の全組み合わせを生成
        combos = list(combinations(sorted(box_horses), 3))
        
        total_odds = 0.0
        valid_count = 0
        
        for combo in combos:
            # オッズ辞書のキーは昇順タプルで格納されている
            if combo in sanrenpuku_odds:
                total_odds += sanrenpuku_odds[combo]
                valid_count += 1
        
        if valid_count > 0:
            avg_odds = total_odds / valid_count
            print(f"    BOX期待値: {avg_odds:.2f}倍 ({valid_count}/{len(combos)}組合せ)")
            return avg_odds
        else:
            print(f"    [X] No valid trifecta odds found for BOX")
            return 0.0
    
    def update_report_with_odds(self, race_id_map: Dict[str, str]):
        """レポート内容をオッズで更新"""
        print("\nUpdating report with latest odds...")
        
        updated_content = self.report_content
        update_count = 0
        
        for key, race_id in race_id_map.items():
            data = self.race_data[key]
            
            # レート制限
            time.sleep(1)
            
            # オッズ取得
            odds_data = self.fetch_latest_odds(race_id)
            if not odds_data:
                continue
            
            # 単勝オッズの更新
            if 'tan' in odds_data and odds_data['tan']:
                for horse_num in data['horse_numbers']:
                    if horse_num in odds_data['tan']:
                        new_odds = odds_data['tan'][horse_num]
                        
                        # レポート内の該当箇所を更新（テーブル形式）
                        # | 4 | ギンケイ | 24.6% | 118.4倍 | -> | 4 | ギンケイ | 24.6% | 3.5倍 |
                        pattern = rf'(\|\s*{horse_num}\s*\|[^|]+\|[^|]+\|)\s*[\d\.]+倍'
                        replacement = rf'\1 {new_odds:.1f}倍'
                        
                        # この馬番が含まれるセクションのみを対象に更新
                        # 複数レースの同じ馬番が誤爆しないよう、セクション単位で処理
                        section_start = updated_content.find(f"### {key}:")
                        if section_start != -1:
                            section_end = updated_content.find("\n---\n", section_start)
                            if section_end == -1:
                                section_end = len(updated_content)
                            
                            section = updated_content[section_start:section_end]
                            updated_section = re.sub(pattern, replacement, section)
                            updated_content = updated_content[:section_start] + updated_section + updated_content[section_end:]
                            update_count += 1
            
            # 三連複BOX期待値の更新
            if data['box_horses'] and 'sanrenpuku' in odds_data and odds_data['sanrenpuku']:
                expected_value = self.calculate_box_expected_value(
                    data['box_horses'], 
                    odds_data['sanrenpuku']
                )
                
                if expected_value > 0:
                    # 既存の期待値を更新（例: 平均期待値7.29倍 -> 平均期待値12.34倍）
                    section_start = updated_content.find(f"### {key}:")
                    if section_start != -1:
                        section_end = updated_content.find("\n---\n", section_start)
                        if section_end == -1:
                            section_end = len(updated_content)
                        
                        section = updated_content[section_start:section_end]
                        pattern = r'平均期待値[\d\.]+倍'
                        replacement = f'平均期待値{expected_value:.2f}倍'
                        updated_section = re.sub(pattern, replacement, section)
                        updated_content = updated_content[:section_start] + updated_section + updated_content[section_end:]
        
        self.report_content = updated_content
        print(f"\nUpdated {update_count} odds entries.")
    
    def save_report(self):
        """更新したレポートを保存"""
        print(f"\nSaving updated report to: {self.report_path}")
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(self.report_content)
        print("[OK] Report saved successfully.")
    
    def run(self):
        """全処理を実行"""
        self.load_report()
        self.parse_report()
        race_id_map = self.build_race_ids()
        
        print(f"\nRace ID mapping:")
        for key, race_id in race_id_map.items():
            print(f"  {key} -> {race_id}")
        
        self.update_report_with_odds(race_id_map)
        self.save_report()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_report_odds.py <report_file>")
        print("Example: python scripts/update_report_odds.py prediction_report_20260131.md")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    if not os.path.exists(report_path):
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)
    
    updater = ReportOddsUpdater(report_path)
    updater.run()
    
    print("\n[OK] All done!")


if __name__ == "__main__":
    main()
