import os
import sys
import re
import pickle
import pandas as pd
from datetime import datetime

# プロジェクトルートの設定
# scripts/prediction/surgical_fix_report_v2.py から見て2段階上
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.data_loader import fetch_and_process_race_data
from modules.scraping import Odds
from modules.betting_allocator import BettingAllocator
import app

# オッズ取得をスキップ（高速化のため。実際の購入ではないため過去オッズで代用）
original_odds_scrape = Odds.scrape
Odds.scrape = lambda x: {}

def update_report_with_details():
    report_path = os.path.join(PROJECT_ROOT, "reports", "prediction_20260221_all_models.md")
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    app.load_model('lgbm')
    app.load_model('ltr')
    app.load_model('stacking')

    new_lines = []
    i = 0
    current_race_id = None
    
    # モデルと戦略の組み合わせ
    model_strategies = [
        ('stacking', 'box4_sanrenpuku'),
        ('stacking', 'ranking_anchor'),
        ('stacking', 'wide_nagashi'),
        ('lgbm', 'box4_sanrenpuku'),
        ('lgbm', 'ranking_anchor'),
        ('lgbm', 'wide_nagashi'),
        ('ltr', 'box4_sanrenpuku'),
        ('ltr', 'ranking_anchor'),
        ('ltr', 'wide_nagashi')
    ]

    budget = 5000

    while i < len(lines):
        line = lines[i]
        
        # Race IDの抽出 (例: ## ダイヤモンドステークス (202605010711))
        race_match = re.search(r'\((\d{12})\)', line)
        if race_match:
            current_race_id = race_match.group(1)
            print(f"Processing Race: {current_race_id}")
            new_lines.append(line)
            i += 1
            continue

        # モデル・戦略セクションの開始を検知
        # 例: ### モデル: ltr / 戦略: ranking_anchor (自信度: S)
        strat_match = re.search(r'### モデル: (\w+) / 戦略: (\w+)', line)
        if strat_match and current_race_id:
            model_type = strat_match.group(1)
            strategy = strat_match.group(2)
            
            # 自信度などは元の行を維持
            new_lines.append(line)
            i += 1
            
            # 推論実行（キャッシュ活用）
            try:
                # データの取得（モック化されていれば速い）
                df, _, _ = fetch_and_process_race_data(current_race_id)
                
                # app.pyのロジックを模倣して予測と配分を取得
                # run_prediction_logic は jsonify を返すので内部処理を流用
                model = app.get_model(model_type)
                feature_names = model.feature_names
                
                # 特徴量エンジニアリング（app.pyのload_modelでロードされたものを使用）
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
                
                # LTRスケーリング
                if model_type == 'ltr':
                    max_p = max([r['probability'] for r in results]) if results else 1.0
                    min_p = min([r['probability'] for r in results]) if results else 0.0
                    range_p = max_p - min_p if max_p != min_p else 1.0
                    for r in results:
                        r['probability'] = (r['probability'] - min_p) / range_p
                
                # 配分
                df_preds = pd.DataFrame(results)
                recommendations = BettingAllocator.allocate_budget(df_preds, budget, strategy=strategy)
                
                # 既存の推奨行をスキップ（次のセクションまたは区切り線まで）
                while i < len(lines) and not lines[i].startswith('###') and not lines[i].startswith('---') and not lines[i].startswith('##'):
                    i += 1
                
                # 新しい推奨を出力
                if recommendations:
                    for rec in recommendations:
                        # 例: - ワイド 流し (軸:3 - 相手:6,10,11,13): 2000円 (4点)
                        bet_str = f"- {rec['bet_type']} {rec['method']} ({rec['combination']}): {rec['total_amount']}円 ({rec['points']}点)\n"
                        new_lines.append(bet_str)
                else:
                    new_lines.append("- (推奨なし: 条件不一致)\n")
                
                new_lines.append("\n")
                continue
            except Exception as e:
                print(f"Error in race {current_race_id} {model_type}: {e}")
                new_lines.append(line)
        
        else:
            new_lines.append(line)
        
        i += 1

    # 行末の重複改行などを整理して書き出し
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Successfully updated: {report_path}")

if __name__ == "__main__":
    update_report_with_details()
