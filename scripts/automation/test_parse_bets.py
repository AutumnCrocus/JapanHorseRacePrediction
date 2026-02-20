"""パースのテスト確認"""
import re
import os
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
REPORT_FILE = os.path.join(PROJECT_ROOT, 'reports', 'prediction_20260221_all_models.md')
BEST_JSON = os.path.join(PROJECT_ROOT, 'reports', 'best_model_strategy.json')

with open(BEST_JSON, 'r', encoding='utf-8') as f:
    best = json.load(f)
target_model = best['model']
target_strategy = best['strategy']
target_header = f"モデル: {target_model} / 戦略: {target_strategy}"
print(f"ターゲット: {target_model} / {target_strategy}")

bets_by_race = {}
current_race_id = None
in_target_section = False

with open(REPORT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip()
    m_race = re.search(r'## (東京|阪神|小倉)(\d+)R\s*\((\d+)\)', line)
    if m_race:
        current_race_id = m_race.group(3)
        bets_by_race[current_race_id] = []
        in_target_section = False
        continue

    if line.startswith('###') and target_header in line:
        in_target_section = True
        continue
    elif line.startswith('###'):
        in_target_section = False
        continue

    if in_target_section and current_race_id and line.startswith('- '):
        m = re.match(
            r'- (単勝|複勝|馬連|ワイド|馬単|3連複|3連単)\s+(BOX|流し|SINGLE|Formation)\s+\((.+?)\):\s+(\d+)円',
            line
        )
        if m:
            bet_type = m.group(1)
            method_raw = m.group(2)
            combo_str = m.group(3)
            amount = int(m.group(4))
            horse_str = re.sub(r'\s*BOX\s*', '', combo_str)
            horses_raw = re.split(r'[,\s]+', horse_str.strip())
            horses = [int(h) for h in horses_raw if h.strip().isdigit()]
            bets_by_race[current_race_id].append({
                'type': bet_type, 'method': 'BOX', 'horses': horses,
                'total_amount': amount, 'raw_line': line.strip()
            })

bets_by_race = {k: v for k, v in bets_by_race.items() if v}
print(f"パース済みレース数: {len(bets_by_race)}")
for rid, blist in list(bets_by_race.items())[:10]:
    print(f"  {rid}: {[b['raw_line'] for b in blist]}")
