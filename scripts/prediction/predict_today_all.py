
import os
import sys
import re
import time
import requests
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor
from modules.betting_allocator import BettingAllocator

TARGET_DATE = '20260201'
BUDGET = 5000
MIN_RACE = 1  # デフォルトは全レース

def get_race_ids(date_str):
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    print(f"Fetching race IDs from: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = 'EUC-JP'
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        links = soup.find_all('a', href=True)
        race_ids = []
        for link in links:
            href = link['href']
            match = re.search(r'race_id=(\d+)', href)
            if match:
                rid = match.group(1)
                if rid.startswith(date_str[:4]):
                    race_ids.append(rid)
        
        race_ids = sorted(list(set(race_ids)))
        print(f"Found {len(race_ids)} races.")
        
        if not race_ids:
            print("DEBUG: No race IDs found via list page. Starting brute-force scan...")
            race_ids = scan_race_ids_brute_force(date_str)
            
        return race_ids
    except Exception as e:
        print(f"Error fetching race IDs: {e}")
        return []

def scan_race_ids_brute_force(date_str):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    year = date_str[:4]
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    target_date_jp = f"{year}年{month}月{day}日"
    print(f"Target Date: {target_date_jp}")
    
    places = [5, 8, 10] # Tokyo, Kyoto, Kokura
    kais = range(1, 4)
    days = range(1, 13)
    
    keys = []
    for p in places:
        for k in kais:
            for d in days:
                keys.append(f"{year}{p:02}{k:02}{d:02}")
    
    active_keys = []
    
    def check_key(key):
        rid = key + "01"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.encoding = 'EUC-JP'
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    print(f"DEBUG: Found match {rid}")
                    return key
        except: pass
        return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in tqdm(as_completed(futures), total=len(keys), desc="Scanning"):
            result = future.result()
            if result:
                active_keys.append(result)
                
    final_ids = []
    for key in active_keys:
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
            
    return final_ids

def load_prediction_pipeline():
    print("Loading models...")
    if os.path.exists(os.path.join(MODEL_DIR, 'production_model.pkl')):
        model = HorseRaceModel()
        model.load(os.path.join(MODEL_DIR, 'production_model.pkl'))
    elif os.path.exists(os.path.join(MODEL_DIR, 'model_lgbm_0.pkl')):
        model = EnsembleModel()
        model.load(MODEL_DIR)
    else:
        model = HorseRaceModel()
        model.load()

    import pickle
    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def process_race(race_id, predictor, budget=BUDGET, horse_results_db=None, peds_db=None):
    print(f"\nProcessing Race ID: {race_id}")
    
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        return None

    # Fix lists
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)

    df_shutuba['date'] = pd.to_datetime(f"{TARGET_DATE[:4]}-{TARGET_DATE[4:6]}-{TARGET_DATE[6:8]}")

    odds_data = Odds.scrape(race_id)
    if odds_data and 'tan' in odds_data:
        for idx, row in df_shutuba.iterrows():
            try:
                umaban = int(row['馬番'])
                if umaban in odds_data['tan']:
                     df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
            except: pass

    # Preprocess
    df_processed = predictor.processor.process_results(df_shutuba)
    
    # Weight Fill
    if '体重' in df_processed.columns:
        mean_weight = df_processed['体重'].mean()
        if pd.isna(mean_weight): mean_weight = 470.0
        df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
            
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    # Features
    if horse_results_db is not None:
         df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
         df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed)
    
    if peds_db is not None:
        df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_db)
        
    df_processed = predictor.engineer.add_odds_features(df_processed)
    
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam', '枠番', '馬番']
    
    # Simple encode or use processor logic? Using processor logic
    # But wait, app.py has explicit categorical handling for LightGBM.
    # predictor.model.predict handles X preparation if using standard format?
    # No, we need to replicate app.py logic roughly or trust predictor logic.
    # predict_tomorrow.py implements explicit column matching manually.
    
    feature_names = predictor.model.feature_names
    X = pd.DataFrame(index=df_processed.index)
    for col in feature_names:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0
            
    # Categorical handling - Important for LightGBM
    try:
        debug_info = predictor.model.debug_info()
        model_cats = debug_info.get('pandas_categorical', [])
        if len(model_cats) >= 2:
             if '枠番' in X.columns:
                 # Ensure it's treated as categorical with specific categories
                 cat_type = pd.CategoricalDtype(categories=model_cats[0], ordered=False)
                 X['枠番'] = X['枠番'].astype(cat_type)
             if '馬番' in X.columns:
                 cat_type = pd.CategoricalDtype(categories=model_cats[1], ordered=False)
                 X['馬番'] = X['馬番'].astype(cat_type)
    except Exception as e:
        print(f"Warning in categorical conversion: {e}")

    
    numeric_X = X.select_dtypes(include=[np.number])
    X[numeric_X.columns] = numeric_X.fillna(numeric_X.median())
    # X = X.fillna(0)  # Removed to avoid TypeError on Categorical columns
    
    # Ensure no object columns remain (LightGBM doesn't like object)
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                # Try converting to numeric first
                X[col] = pd.to_numeric(X[col])
            except (ValueError, TypeError):
                # If fail, fill with 0 (treat as unknown/missing) instead of converting to category
                # This avoids 'categorical_feature do not match' error when we don't have the original encoder
                print(f"Warning: Filling object column '{col}' with 0 due to missing encoder")
                X[col] = 0
                
    probs = predictor.model.predict(X)
    
    results_df = df_shutuba.copy()
    results_df['probability'] = probs
    results_df['horse_number'] = results_df['馬番'].astype(int)
    results_df['horse_name'] = results_df['馬名']
    
    results_df['odds'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(0)
    results_df['expected_value'] = results_df['probability'] * results_df['odds']
    
    recommendations = BettingAllocator.allocate_budget(
        results_df, 
        budget=budget, 
        odds_data=odds_data
    )
    
    # Confidence Logic (Matches app.py)
    results_sorted = results_df.sort_values('probability', ascending=False)
    confidence_level = 'D'
    if not results_sorted.empty:
        top_prob = results_sorted.iloc[0]['probability']
        top_ev = results_sorted.iloc[0]['expected_value']
        
        if top_prob >= 0.5 or top_ev >= 1.5:
            confidence_level = 'S'
        elif top_prob >= 0.4 or top_ev >= 1.2:
            confidence_level = 'A'
        elif top_prob >= 0.3 or top_ev >= 1.0:
            confidence_level = 'B'
        elif top_prob >= 0.2:
            confidence_level = 'C'
        else:
            confidence_level = 'D'
    
    race_info = {
        'race_id': race_id,
        'race_name': df_shutuba.iloc[0].get('レース名', 'Unknown Race').strip(),
        'race_time': df_shutuba.attrs.get('race_data01', ''),
        'venue': get_venue_name(race_id),
        'confidence': confidence_level
    }
    
    return {
        'info': race_info,
        'recommendations': recommendations,
        'predictions': results_sorted.head(5)[['馬番', '馬名', 'probability', '単勝', 'expected_value']].to_dict('records')
    }

def get_venue_name(race_id):
    place_code = race_id[4:6]
    codes = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    return codes.get(place_code, 'Unknown')

def load_historical_data():
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
            
        try:
            ped_scores = {
                'speed': ['ディープインパクト', 'ロードカナロア', 'サクラバクシンオー', 'ダイワメジャー', 'キングカメハメハ'],
                'stamina': ['ハーツクライ', 'オルフェーヴル', 'ゴールドシップ', 'ステイゴールド', 'エピファネイア'],
                'dirt': ['ヘニーヒューズ', 'シニスターミニスター', 'ゴールドアリュール', 'パイロ', 'クロフネ']
            }
            peds_str_series = peds.fillna('').astype(str).agg(' '.join, axis=1)
            for cat, sires in ped_scores.items():
                def count_s(text):
                    c = 0
                    for s in sires:
                        if s in text: c += 1
                    return c
                peds[f'peds_score_{cat}'] = peds_str_series.apply(lambda x: count_s(x))
        except: pass
            
    return horse_results, peds

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='競馬予想レポート生成')
    parser.add_argument('--min-race', type=int, default=MIN_RACE, 
                        help='最小レース番号（例: 9で9R以降のみ対象）')
    parser.add_argument('--budget', type=int, default=BUDGET, 
                        help='1レースあたりの予算（円）')
    args = parser.parse_args()
    
    min_race = args.min_race
    budget = args.budget
    
    print(f"Start Prediction for {TARGET_DATE} (Budget: {budget} JPY/race, Min Race: {min_race}R)")
    race_ids = get_race_ids(TARGET_DATE)
    
    if not race_ids:
        print("No races found.")
        return

    # 最小レース番号でフィルタリング
    filtered_race_ids = []
    for rid in race_ids:
        race_num = int(rid[-2:])  # レースIDの末尾2桁がレース番号
        if race_num >= min_race:
            filtered_race_ids.append(rid)
    
    print(f"Filtered to {len(filtered_race_ids)} races (Race {min_race}R and above)")
    
    if not filtered_race_ids:
        print(f"No races found for {min_race}R and above.")
        return

    predictor = load_prediction_pipeline()
    horse_results, peds = load_historical_data()
    
    all_results = []
    
    for rid in tqdm(filtered_race_ids, desc="Processing Races"):
        try:
            res = process_race(rid, predictor, budget=budget, horse_results_db=horse_results, peds_db=peds)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\nError processing {rid}: {e}")
            
    # Output Report
    suffix = f"_from{min_race}R" if min_race > 1 else ""
    report_file = os.path.join(os.getcwd(), f"prediction_today_{TARGET_DATE}{suffix}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬予想レポート ({TARGET_DATE[0:4]}/{TARGET_DATE[4:6]}/{TARGET_DATE[6:8]})\n")
        f.write(f"予算: 各レース {budget}円")
        if min_race > 1:
            f.write(f" (対象: {min_race}R以降のみ)")
        f.write("\n\n")
        
        venue_groups = {}
        for r in all_results:
            v = r['info']['venue']
            if v not in venue_groups: venue_groups[v] = []
            venue_groups[v].append(r)
            
        for venue, races in venue_groups.items():
            f.write(f"## {venue}開催\n")
            for race in races:
                info = race['info']
                rid = info['race_id']
                r_num = rid[-2:]
                
                conf = info['confidence']
                conf_mark = "🔥" if conf in ['S', 'A'] else ""
                
                f.write(f"### {venue}{r_num}R (ID:{rid}) {info['race_name']} - 推奨度: {conf} {conf_mark}\n")
                if info['race_time']:
                    f.write(f"_{info['race_time']}_\n\n")
                
                recs = race['recommendations']
                if not recs:
                    f.write("> **推奨買い目なし** (条件不適合)\n")
                else:
                    total_buy = sum([r['total_amount'] for r in recs if 'total_amount' in r] or [r['unit_amount'] for r in recs if 'unit_amount' in r]) # simple sum
                    
                    f.write("#### 🎯 推奨買い目\n")
                    for rec in recs:
                        method = rec.get('bet_type', '') + " " + rec.get('method', '')
                        combo = rec.get('combination', '') or f"馬番:{rec.get('horse_numbers')}"
                        amount = rec.get('total_amount', 0)
                        if amount == 0: amount = rec.get('unit_amount', 0)
                        
                        f.write(f"- **{method}**: {combo} ({amount}円)\n")
                        if 'reason' in rec:
                            f.write(f"  - _{rec['reason']}_\n")
                            
                f.write("\n#### 📊 AI注目馬\n")
                f.write("| 馬番 | 馬名 | 勝率 | オッズ | 期待値 |\n")
                f.write("| :---: | :--- | :---: | :---: | :---: |\n")
                preds = race['predictions']
                for p in preds:
                    odds = p['単勝']
                    odds_str = f"{odds}" if odds else "-"
                    ev = p.get('expected_value', 0)
                    f.write(f"| {p['馬番']} | {p['馬名']} | {p['probability']:.1%} | {odds_str} | {ev:.2f} |\n")
                f.write("\n---\n")
                
    print(f"\nReport generated: {report_file}")

if __name__ == "__main__":
    main()
