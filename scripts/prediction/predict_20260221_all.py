import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import app
from app import load_model, run_prediction_logic, MODELS
from modules.scraping import Shutuba, Odds, get_race_date_info
from modules.data_loader import fetch_and_process_race_data

def discover_race_ids(target_date):
    """
    指定された日付の有効なレースIDの一覧を取得する
    YYYYMMDD -> 10桁 (YYYY + 会場2桁 + 開催2桁 + 日目2桁) + レース2桁
    例: 20260221 -> 2026 05 01 07 01 (東京7日目1R)
    """
    print(f"{target_date} のレースIDを探索中 (最適化版)...")
    # 2026/02/21 (土) の開催会場を絞り込む (東京:05, 阪神:09, 小倉:10 が一般的)
    venues = ["05", "09", "10"] 
    valid_ids = []
    
    # 開催(kai)と日目(day)の組み合わせを探索
    for v in venues:
        for kai in range(1, 3): # 1-2回
            for day in range(1, 9): # 1-8日目
                # レースは11Rや12Rがメインなので、とりあえず12Rが存在するかチェック
                race_id_prefix = f"2026{v}{kai:02d}{day:02d}"
                test_race_id = f"{race_id_prefix}11" # 11Rで代表チェック
                
                try:
                    info = Shutuba.scrape(test_race_id)
                    if not info.empty:
                        # 取得できた場合、日付が一致するか確認
                        date_info = get_race_date_info(test_race_id)
                        # '2026年2月21日' 形式
                        target_str = f"{target_date[:4]}年{int(target_date[4:6])}月{int(target_date[6:8])}日"
                        if info.attrs.get('race_name') and target_str in date_info.get('date', ''):
                            print(f"会場 {v} (第{kai}回{day}日目) を発見しました。")
                            for r_num in range(1, 13):
                                valid_ids.append(f"{race_id_prefix}{r_num:02d}")
                except:
                    continue
    return valid_ids

def predict_all_races_20260221():
    target_date = "20260221"
    budget_per_combination = 5000
    
    # 利用可能なモデルと戦略の定義
    models_to_test = ['stacking', 'ltr', 'lgbm']
    strategies_to_test = ['box4_sanrenpuku', 'ranking_anchor', 'wide_nagashi']
    
    print(f"=== {target_date} 全レース予測開始 (マルチモデル/戦略探索) ===")
    
    # 1. 有効なレースIDを取得
    race_ids = discover_race_ids(target_date)
    if not race_ids:
        print("有効なレースが見つかりませんでした。")
        return

    print(f"合計 {len(race_ids)} レースの予測を実行します。")

    # 2. モデルのロード
    for m_type in models_to_test:
        print(f"Loading model: {m_type}")
        load_model(m_type)

    all_results = []
    
    # 3. 予測ループ
    from flask import Flask
    test_app = Flask(__name__)
    
    for rid in race_ids:
        try:
            print(f"\n>>>> 処理中: {rid} <<<<")
            # 出馬表・特徴量取得 (重複ロードを防ぐため内部でprocessor等は管理)
            # app.pyのget_model(m_type)で得られるprocessor/engineerを使用
            m_lgbm = MODELS.get('lgbm')
            if not m_lgbm: load_model('lgbm'); m_lgbm = MODELS.get('lgbm')
            
            from app import PROCESSORS, ENGINEERS
            df = fetch_and_process_race_data(rid, PROCESSORS['lgbm'], ENGINEERS['lgbm'], 
                                             app.bias_map, app.jockey_stats, 
                                             app.horse_results, app.peds)
            
            if df is None or df.empty:
                print(f"Skip {rid}: データ取得失敗")
                continue
            
            race_name = df.attrs.get('race_name', '不明')
            print(f"レース確定: {race_name}")
            
            race_summary = {
                'race_id': rid,
                'race_name': race_name,
                'combinations': []
            }
            
            for m_type in models_to_test:
                for strategy in strategies_to_test:
                    with test_app.app_context():
                        res = run_prediction_logic(df, race_name, "Info", 
                                                 race_id=rid, budget=budget_per_combination, 
                                                 strategy=strategy, model_type=m_type)
                        data = res.get_json()
                        
                        if data.get('success'):
                            recs = data.get('recommendations', [])
                            if recs:
                                race_summary['combinations'].append({
                                    'model': m_type,
                                    'strategy': strategy,
                                    'recs': recs,
                                    'confidence': data.get('confidence_level')
                                })
            
            if race_summary['combinations']:
                all_results.append(race_summary)
                
        except Exception as e:
            print(f"Error processing {rid}: {e}")
            continue

    # 4. 結果の保存とレポート作成 (Markdown)
    report_file = f"reports/prediction_{target_date}_all_models.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 2026/02/21 全レース予想レポート (マルチモデル・戦略)\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")
        f.write(f"- 予算上限: {budget_per_combination}円 / 組み合わせ\n")
        f.write(f"- 使用モデル: {', '.join(models_to_test)}\n")
        f.write(f"- 使用戦略: {', '.join(strategies_to_test)}\n\n")
        
        for r in all_results:
            f.write(f"## {r['race_name']} ({r['race_id']})\n\n")
            for c in r['combinations']:
                f.write(f"### モデル: {c['model']} / 戦略: {c['strategy']} (自信度: {c['confidence']})\n")
                if c['recs']:
                    for rec in c['recs']:
                        amount_str = f" ({rec['amount']}円)" if 'amount' in rec else ""
                        f.write(f"- {rec['method']}: {rec['horse_numbers']}{amount_str}\n")
                f.write("\n")
            f.write("---\n\n")
    
    print(f"=== 予測完了: {report_file} ===")

if __name__ == "__main__":
    # Flaskのコンテキストをシミュレートするためにモックが必要な場合があるが、
    # 一旦実行を試みる
    predict_all_races_20260221()
