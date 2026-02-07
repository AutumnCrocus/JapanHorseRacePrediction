
import os
import sys
import re
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor
from modules.betting_allocator import BettingAllocator

def get_race_ids_20260208():
    """
    2026/02/08のレースIDリストを生成する
    東京(05): 1回4日, 京都(08): 2回4日, 小倉(10): 1回6日
    """
    active_keys = ['2026050104', '2026080204', '2026100106']
    final_ids = []
    for key in active_keys:
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
    return final_ids

def load_prediction_pipeline():
    """モデルと前処理パイプラインをロード"""
    print("Loading models...")
    
    # Check for ensemble model first
    if os.path.exists(os.path.join(MODEL_DIR, 'model_lgbm_0.pkl')):
        model = EnsembleModel()
        model.load(MODEL_DIR)
        print("Ensemble model loaded.")
    else:
        model = HorseRaceModel()
        model.load()
        print("Single model loaded.")

    # Processor / Engineer
    import pickle
    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def process_race(race_id, predictor, budget=10000, horse_results_db=None, peds_db=None):
    """1レース分の処理を実行 (Balance戦略, 予算10000円)"""
    print(f"\nProcessing Race ID: {race_id}")
    
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        print("  Failed to fetch shutuba data.")
        return None

    # Cleaning: Validate and fix list values in DataFrame
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)

    # 予測用の日付（当日）
    df_shutuba['date'] = pd.to_datetime('2026-02-08')

    # 2. オッズ取得
    odds_data = Odds.scrape(race_id)
    if odds_data and 'tan' in odds_data:
        tan_odds = odds_data['tan']
        def update_odds_val(row):
            try:
                u = int(row['馬番'])
                val = tan_odds.get(u)
                if val is not None: return val
                return row['単勝']
            except:
                return row['単勝']
        
        df_shutuba['単勝'] = df_shutuba.apply(update_odds_val, axis=1)

    # 3. 前処理 & 特徴量生成
    df_processed = predictor.processor.process_results(df_shutuba)
    
    # 馬体重補完
    if '体重' in df_processed.columns:
        mean_weight = df_processed['体重'].mean()
        if pd.isna(mean_weight): mean_weight = 470.0
        df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
            
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    # 履歴特徴量追加
    if horse_results_db is not None:
         # Filter horse_results to relevant horses (Optimization)
         if 'horse_id' in df_processed.columns:
             target_ids = df_processed['horse_id'].astype(str).unique()
             
             # Create a copy or view
             hr_subset = None
             
             # Check if horse_id is in columns
             if 'horse_id' in horse_results_db.columns:
                 hr_subset = horse_results_db[horse_results_db['horse_id'].astype(str).isin(target_ids)]
             # Check index
             elif horse_results_db.index.name == 'horse_id' or 'horse_id' in horse_results_db.index.names:
                 # Filter by index
                 common_ids = horse_results_db.index.intersection(target_ids)
                 if not common_ids.empty:
                     hr_subset = horse_results_db.loc[common_ids]
             # Check if index is unnamed but acts as horse_id
             else:
                 # Try to filter by index values
                 # Convert index to str and check
                 idx_str = horse_results_db.index.astype(str)
                 mask = idx_str.isin(target_ids)
                 hr_subset = horse_results_db[mask]

             if hr_subset is not None and not hr_subset.empty:
                 df_processed = predictor.engineer.add_horse_history_features(df_processed, hr_subset)
                 df_processed = predictor.engineer.add_course_suitability_features(df_processed, hr_subset)
             else:
                 df_processed = predictor.engineer.add_horse_history_features(df_processed, pd.DataFrame()) 
                 df_processed = predictor.engineer.add_course_suitability_features(df_processed, pd.DataFrame())
         else:
             df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
             df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed)
    
    if peds_db is not None:
        # Filter peds_db to relevant horses only (Optimization)
        if 'horse_id' in df_processed.columns:
            relevant_ids = df_processed['horse_id'].astype(str).unique()
            peds_subset = peds_db[peds_db.index.astype(str).isin(relevant_ids)]
            df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_subset)
        else:
             df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_db)
        
    df_processed = predictor.engineer.add_odds_features(df_processed)
    
    # カテゴリエンコード
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df_processed.columns]
    df_processed = predictor.processor.encode_categorical(df_processed, cat_cols)
    
    # 特徴量選択 & 欠損埋め
    feature_names = predictor.model.feature_names
    X = pd.DataFrame(index=df_processed.index)
    for col in feature_names:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0
    
    numeric_X = X.select_dtypes(include=[np.number])
    X[numeric_X.columns] = numeric_X.fillna(numeric_X.median())
    X = X.fillna(0)
    
    if predictor.processor.scaler:
        X = predictor.processor.transform_scale(X)

    # 4. 予測
    probs = predictor.model.predict(X)
    
    results_df = df_shutuba.copy()
    results_df['probability'] = probs
    results_df['horse_number'] = results_df['馬番'].astype(int)
    results_df['horse_name'] = results_df['馬名']
    results_df['odds'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(0)
    results_df['expected_value'] = results_df['probability'] * results_df['odds']
    
    # 5. 予算配分 (Balance Strategy)
    recommendations = BettingAllocator.allocate_budget(
        results_df, 
        budget=budget, 
        odds_data=odds_data,
        strategy='balance' # CHANGED to 'balance'
    )
    
    race_info = {
        'race_id': race_id,
        'race_name': df_shutuba.attrs.get('race_name', 'Race ' + race_id[-2:]),
        'race_time': df_shutuba.attrs.get('race_time', ''),
        'venue': get_venue_name(race_id)
    }
    
    return {
        'info': race_info,
        'recommendations': recommendations,
        'predictions': results_df.sort_values('probability', ascending=False).head(5)[['馬番', '馬名', 'probability', '単勝']].to_dict('records')
    }

def get_venue_name(race_id):
    place_code = race_id[4:6]
    codes = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}
    return codes.get(place_code, 'Unknown')

def load_historical_data():
    """過去データをロード"""
    import pickle
    print("Loading historical data...")
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    horse_results = None
    peds = None
    
    if os.path.exists(hr_path):
        with open(hr_path, 'rb') as f:
            horse_results = pickle.load(f)
    if os.path.exists(peds_path):
        with open(peds_path, 'rb') as f:
            peds = pickle.load(f)
    return horse_results, peds

def main():
    date_str = '20260208'
    race_ids = get_race_ids_20260208()
    
    predictor = load_prediction_pipeline()
    horse_results, peds = load_historical_data()
    
    import socket
    socket.setdefaulttimeout(30)

    all_results = []
    total_races = len(race_ids)
    
    # Budget set to 10,000 yen
    BUDGET_PER_RACE = 10000
    
    for i, rid in enumerate(race_ids):
        print(f"Processing Race {i+1}/{total_races}: {rid}")
        try:
            res = process_race(rid, predictor, budget=BUDGET_PER_RACE, horse_results_db=horse_results, peds_db=peds)
            if res:
                all_results.append(res)
            time.sleep(1) 
        except Exception as e:
            print(f"Error processing {rid}: {e}")
            import traceback
            traceback.print_exc()
            
    report_file = f"prediction_report_{date_str}_balance_10000.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬予想レポート ({date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]}) - バランス戦略 (予算1万円)\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("使用戦略: **バランス型 (Balance)**\n")
        f.write("※馬体重は平均値等で補完しています。\n\n")
        
        venue_groups = {}
        for r in all_results:
            v = r['info']['venue']
            if v not in venue_groups: venue_groups[v] = []
            venue_groups[v].append(r)
            
        for venue, races in venue_groups.items():
            f.write(f"## {venue}開催\n")
            for race in races:
                info = race['info']
                r_num = info['race_id'][-2:]
                f.write(f"### {venue}{r_num}R\n")
                
                f.write(f"#### 推奨買い目 (予算{BUDGET_PER_RACE:,}円)\n")
                recs = race['recommendations']
                if not recs:
                    f.write("- 推奨なし\n")
                else:
                    for rec in recs:
                        f.write(f"- **{rec['bet_type']} {rec['method']}**: {rec['combination']} ({rec['total_amount']}円)\n")
                        f.write(f"  - 理由: {rec['reason']}\n")
                
                f.write("\n#### AI注目馬 (Top 5)\n")
                f.write("| 馬番 | 馬名 | 勝率予測 | 単勝オッズ |\n| :---: | :--- | :---: | :---: |\n")
                for p in race['predictions']:
                    odds_str = f"{p['単勝']}倍" if p['単勝'] else "-"
                    f.write(f"| {p['馬番']} | {p['馬名']} | {p['probability']:.1%} | {odds_str} |\n")
                f.write("\n---\n")
                
    print(f"\nReport generated: {report_file}")

if __name__ == "__main__":
    main()
