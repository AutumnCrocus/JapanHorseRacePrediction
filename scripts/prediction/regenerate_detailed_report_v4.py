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
    
    app.load_model('lgbm')
    app.load_model('ltr')
    app.load_model('stacking')

    prediction_date = "2026/02/21"
    # 会場リスト
    venue_map = {"05": "東京", "09": "阪神", "10": "小倉"}
    budget = 5000

    output = f"# 競馬予想レポート: {prediction_date}\n\n"
    output += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += "予算設定: 1レースあたり 5000円\n\n"

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

    for venue_id, venue_name in venue_map.items():
        output += f"# {venue_name}競馬場\n\n"
        
        if venue_id == "05": kai, day = "01", "07"
        elif venue_id == "09": kai, day = "01", "01"
        elif venue_id == "10": kai, day = "02", "01"
        
        for race_num in range(1, 13):
            race_id = f"2026{venue_id}{kai}{day}{race_num:02d}"
            print(f"Generating for Race: {race_id}")
            
            try:
                df_processed = fetch_and_process_race_data(
                    race_id, 
                    app.PROCESSORS.get('lgbm'), 
                    app.ENGINEERS.get('lgbm'), 
                    bias_map=app.bias_map,
                    jockey_stats=app.jockey_stats,
                    horse_results_df=app.horse_results,
                    peds_df=app.peds
                )
                
                if df_processed is None or df_processed.empty:
                    print(f"Skipping {race_id}: No data")
                    continue
                
                output += f"## {venue_name}{race_num}R ({race_id})\n\n"
                
                for model_type, strategy in model_strategies:
                    model = app.get_model(model_type)
                    feature_names = model.feature_names
                    
                    # DeepFMスコアの動的算出 (app.pyのロジックを模倣)
                    if 'deepfm_score' in feature_names and app.DEEPFM_INFERENCE is not None:
                        if 'deepfm_score' not in df_processed.columns:
                            print(f"  Computing DeepFM scores for {model_type}...")
                            try:
                                df_processed['deepfm_score'] = app.DEEPFM_INFERENCE.predict(df_processed)
                            except Exception as de:
                                print(f"  DeepFM error: {de}")
                                df_processed['deepfm_score'] = 0.5

                    # フィーチャーのフィルタリングとパディング
                    X = df_processed.copy()
                    missing_cols = set(feature_names) - set(X.columns)
                    for col in missing_cols:
                        X[col] = 0.0 # デフォルト値
                    
                    X = X[feature_names]
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    
                    probs = model.predict(X)
                    
                    results = []
                    for idx, (_, row) in enumerate(df_processed.iterrows()):
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
                    sorted_preds = df_preds.sort_values('probability', ascending=False)
                    top_prob = sorted_preds.iloc[0]['probability']
                    top_ev = sorted_preds.iloc[0]['expected_value']
                    
                    conf = 'D'
                    if top_prob >= 0.5 or top_ev >= 1.5: conf = 'S'
                    elif top_prob >= 0.4 or top_ev >= 1.2: conf = 'A'
                    elif top_prob >= 0.3 or top_ev >= 1.0: conf = 'B'
                    elif top_prob >= 0.2: conf = 'C'
                    
                    output += f"### モデル: {model_type} / 戦略: {strategy} (自信度: {conf})\n"
                    
                    recommendations = BettingAllocator.allocate_budget(sorted_preds, budget, strategy=strategy)
                    if recommendations:
                        for rec in recommendations:
                            output += f"- {rec['bet_type']} {rec['method']} ({rec['combination']}): {rec['total_amount']}円 ({rec['points']}点)\n"
                    else:
                        output += "- (推奨条件を満たす組み合わせが見つかりませんでした)\n"
                    output += "\n"
                
                output += "---\n\n"
            except Exception as e:
                print(f"Error processing {race_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Detailed report generated successfully: {report_path}")

if __name__ == "__main__":
    generate_full_detailed_report()
