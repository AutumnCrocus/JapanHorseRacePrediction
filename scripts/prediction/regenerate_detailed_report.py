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

# オッズ取得をスキップ
Odds.scrape = lambda x: {}

def generate_full_detailed_report():
    report_path = os.path.join(PROJECT_ROOT, "reports", "prediction_20260221_all_models.md")
    
    # モデルのロード
    app.load_model('lgbm')
    app.load_model('ltr')
    app.load_model('stacking')

    # 日付と会場設定
    prediction_date = "2026/02/21"
    venues = ["05", "09", "10"] # 東京, 阪神, 小倉
    budget = 5000

    output = f"# 競馬予想レポート: {prediction_date}\n\n"
    output += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"予算設定: 1レースあたり {budget}円\n\n"

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

    for venue in venues:
        venue_name = {"05": "東京", "09": "阪神", "10": "小倉"}.get(venue)
        output += f"# {venue_name}競馬場\n\n"
        
        for race_num in range(1, 13):
            race_id = f"2026{venue}01{race_num:02d}"
            print(f"Generating for Race: {race_id}")
            
            try:
                df, race_name, race_info = fetch_and_process_race_data(race_id)
                if df is None or df.empty:
                    continue
                
                output += f"## {race_name} ({race_id})\n"
                output += f"{race_info}\n\n"
                
                for model_type, strategy in model_strategies:
                    model = app.get_model(model_type)
                    feature_names = model.feature_names
                    
                    X = df[feature_names].copy()
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    
                    probs = model.predict(X)
                    
                    results = []
                    for idx, (_, row) in enumerate(df.iterrows()):
                        results.append({
                            'horse_number': int(row.get('馬番', idx+1)),
                            'horse_name': str(row.get('馬名', f"馬{idx+1}")),
                            'probability': float(probs[idx]),
                            'odds': float(row.get('odds', row.get('単勝', 10.0))),
                            'expected_value': float(probs[idx] * float(row.get('odds', row.get('単勝', 10.0))))
                        })
                    
                    if model_type == 'ltr':
                        max_p = max([r['probability'] for r in results]) if results else 1.0
                        min_p = min([r['probability'] for r in results]) if results else 0.0
                        range_p = max_p - min_p if max_p != min_p else 1.0
                        for r in results:
                            r['probability'] = (r['probability'] - min_p) / range_p
                    
                    df_preds = pd.DataFrame(results)
                    # 自信度の計算（app.pyのロジック流用）
                    top_prob = df_preds.sort_values('probability', ascending=False).iloc[0]['probability']
                    top_ev = df_preds.sort_values('probability', ascending=False).iloc[0]['expected_value']
                    
                    conf = 'D'
                    if top_prob >= 0.5 or top_ev >= 1.5: conf = 'S'
                    elif top_prob >= 0.4 or top_ev >= 1.2: conf = 'A'
                    elif top_prob >= 0.3 or top_ev >= 1.0: conf = 'B'
                    elif top_prob >= 0.2: conf = 'C'
                    
                    output += f"### モデル: {model_type} / 戦略: {strategy} (自信度: {conf})\n"
                    
                    recommendations = BettingAllocator.allocate_budget(df_preds, budget, strategy=strategy)
                    if recommendations:
                        for rec in recommendations:
                            output += f"- {rec['bet_type']} {rec['method']} ({rec['combination']}): {rec['total_amount']}円 ({rec['points']}点)\n"
                    else:
                        output += "- (推奨条件を満たす組み合わせが見つかりませんでした)\n"
                    output += "\n"
                
                output += "---\n\n"
            except Exception as e:
                print(f"Error processing {race_id}: {e}")
                continue

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Detailed report generated successfully: {report_path}")

if __name__ == "__main__":
    generate_full_detailed_report()
