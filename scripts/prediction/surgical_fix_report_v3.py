import os
import sys
import re
import pandas as pd
from datetime import datetime

# プロジェクトルートの設定
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.data_loader import fetch_and_process_race_data
from modules.scraping import Odds
from modules.betting_allocator import BettingAllocator
import app

# オッズ取得をスキップ（高速化のため）
Odds.scrape = lambda x: {}

def update_report_with_details():
    report_path = os.path.join(PROJECT_ROOT, "reports", "prediction_20260221_all_models.md")
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    app.load_model('lgbm')
    app.load_model('ltr')
    app.load_model('stacking')

    # レースごとに分割
    # ## ◯◯R (ID) で分割
    race_blocks = re.split(r'(?=\n## .+\(\d{12}\))', content)
    
    new_content = race_blocks[0] # ヘッダー部分
    
    budget = 5000

    for block in race_blocks[1:]:
        # Race IDの抽出
        race_match = re.search(r'\((\d{12})\)', block)
        if not race_match:
            new_content += block
            continue
            
        race_id = race_match.group(1)
        print(f"Processing Race: {race_id}")
        
        # モデル・戦略セクションごとに分割
        # ### モデル: ... で分割
        strat_blocks = re.split(r'(?=\n### モデル: )', block)
        
        race_header = strat_blocks[0]
        new_race_block = race_header
        
        try:
            df, _, _ = fetch_and_process_race_data(race_id)
        except Exception as e:
            print(f"Failed to fetch data for {race_id}: {e}")
            new_content += block
            continue

        for strat_block in strat_blocks[1:]:
            strat_header_match = re.search(r'(### モデル: (\w+) / 戦略: (\w+).*\n)', strat_block)
            if not strat_header_match:
                new_race_block += strat_block
                continue
                
            strat_header = strat_header_match.group(1)
            model_type = strat_header_match.group(2)
            strategy = strat_header_match.group(3)
            
            # 推論実行（モック化されたデータを想定）
            try:
                model = app.get_model(model_type)
                feature_names = model.feature_names
                
                X = df[feature_names].copy()
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
                probs = model.predict(X)
                
                results = []
                for idx, (_, row) in enumerate(df.iterrows()):
                    odds = row.get('odds', row.get('単勝', 10.0))
                    results.append({
                        'horse_number': int(row.get('馬番', idx+1)),
                        'horse_name': str(row.get('馬名', f"馬{idx+1}")),
                        'probability': float(probs[idx]),
                        'odds': float(odds),
                        'expected_value': float(probs[idx] * float(odds))
                    })
                
                if model_type == 'ltr':
                    max_p = max([r['probability'] for r in results]) if results else 1.0
                    min_p = min([r['probability'] for r in results]) if results else 0.0
                    range_p = max_p - min_p if max_p != min_p else 1.0
                    for r in results:
                        r['probability'] = (r['probability'] - min_p) / range_p
                
                df_preds = pd.DataFrame(results)
                recommendations = BettingAllocator.allocate_budget(df_preds, budget, strategy=strategy)
                
                new_race_block += strat_header
                if recommendations:
                    for rec in recommendations:
                        bet_str = f"- {rec['bet_type']} {rec['method']} ({rec['combination']}): {rec['total_amount']}円 ({rec['points']}点)\n"
                        new_race_block += bet_str
                else:
                    new_race_block += "- (推奨条件を満たす組み合わせが見つかりませんでした)\n"
                
                new_race_block += "\n"
                
            except Exception as e:
                print(f"Error in {race_id} / {model_type} / {strategy}: {e}")
                new_race_block += strat_block
                
        new_content += new_race_block

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully updated: {report_path}")

if __name__ == "__main__":
    update_report_with_details()
