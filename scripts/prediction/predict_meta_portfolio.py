
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
from datetime import datetime, timedelta

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === CONFIG ===
DEFAULT_BUDGET_PER_STRATEGY = 5000

def get_target_date(date_arg=None):
    """ターゲット日付を取得 (YYYYMMDD)"""
    if date_arg:
        return date_arg
    
    # デフォルトは明日
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%Y%m%d")

def get_race_ids(date_str):
    """
    指定された日付のレースIDを取得する
    URL: https://race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD
    """
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
    """
    総当たりでその日に開催されるレースIDを特定する
    ID形式: YYYY PP KK DD RR
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    year = date_str[:4]
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    target_date_jp = f"{year}年{month}月{day}日"
    print(f"Target Date: {target_date_jp}")
    
    # 探索範囲
    places = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 全会場候補
    
    kais = range(1, 4)    # 1~3回
    days = range(1, 13)   # 1~12日目
    
    # 候補となる「会場・回・日」のキー (YYYYPPKKDD)
    keys = []
    for p in places:
        for k in kais:
            for d in days:
                keys.append(f"{year}{p:02}{k:02}{d:02}")
    
    active_keys = []
    
    # Cache for known dates (optimization)
    if date_str == '20260201':
         # Found keys: 2026050101 (Tokyo), 2026080201 (Kyoto), 2026100103 (Kokura)
         return [f"2026050101{r:02}" for r in range(1, 13)] + \
                [f"2026080201{r:02}" for r in range(1, 13)] + \
                [f"2026100103{r:02}" for r in range(1, 13)]
    
    if date_str == '20260208':
         # Predicting keys based on schedule pattern (just a guess, likely same venues but next day or next week)
         # If brute force is too slow, I'll rely on the scan.
         pass
    
    def check_key(key):
        # 1RのIDで存在確認
        rid = key + "01"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.encoding = 'EUC-JP'
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    return key
            return None
        except:
            return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in tqdm(as_completed(futures), total=len(keys), desc="Searching"):
            result = future.result()
            if result:
                active_keys.append(result)
                
    final_ids = []
    for key in set(active_keys):
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
            
    return final_ids

def load_prediction_pipeline():
    """モデルと前処理パイプラインをロード"""
    print("Loading models...")
    
    # Ensemble check
    if os.path.exists(os.path.join(MODEL_DIR, 'model_lgbm_0.pkl')):
        model = EnsembleModel()
        model.load(MODEL_DIR)
        print("Ensemble model loaded.")
    else:
        model = HorseRaceModel()
        path = os.path.join(MODEL_DIR, "historical_2010_2024", "model.pkl")
        if not os.path.exists(path):
            path = os.path.join(MODEL_DIR, "model.pkl")
        model.load(path)
        print(f"Single model loaded from {path}")

    import pickle
    # Try historical dir first
    hist_dir = os.path.join(MODEL_DIR, "historical_2010_2024")
    proc_path = os.path.join(hist_dir, 'processor.pkl')
    eng_path = os.path.join(hist_dir, 'engineer.pkl')
    
    if not os.path.exists(proc_path):
        proc_path = os.path.join(MODEL_DIR, 'processor.pkl')
        eng_path = os.path.join(MODEL_DIR, 'engineer.pkl')

    with open(proc_path, 'rb') as f:
        processor = pickle.load(f)
    with open(eng_path, 'rb') as f:
        engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def process_race(race_id, predictor, horse_results_db=None, peds_db=None):
    """1レース分の処理を実行"""
    print(f"\nProcessing Race ID: {race_id}")
    
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        print("  Failed to fetch shutuba data.")
        return None

    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            df_shutuba[col] = df_shutuba[col].apply(lambda x: str(x[0]) if isinstance(x, list) and len(x) > 0 else (str(x) if not isinstance(x, list) else ""))

    # 日付補完
    date_str = race_id[:8] # YYYYMMDD
    try:
        dt = pd.to_datetime(date_str, format='%Y%m%d')
        df_shutuba['date'] = dt
    except:
        df_shutuba['date'] = pd.to_datetime("2026-01-01")

    # 2. オッズ取得
    odds_data = Odds.scrape(race_id)
    if odds_data and 'tan' in odds_data:
        for idx, row in df_shutuba.iterrows():
            try:
                umaban = int(row['馬番'])
                if umaban in odds_data['tan']:
                     df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
            except: pass

    # 3. 前処理 & 特徴量生成
    df_processed = predictor.processor.process_results(df_shutuba)
    
    # 馬体重補完
    if '体重' in df_processed.columns:
        mean_weight = df_processed['体重'].mean()
        if pd.isna(mean_weight): mean_weight = 470.0
        df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
            
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    if horse_results_db is not None:
         df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
         df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed)
    
    if peds_db is not None:
        df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_db)
        
    df_processed = predictor.engineer.add_odds_features(df_processed)
    
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df_processed.columns]
    
    # カテゴリエンコーディング時の未知ラベル対応
    # ProcessorのLabelEncoderはfit済みなので、未知ラベルはエラーになる可能性がある
    # ここでは簡易的にtry-catchするか、Processor側でhandle_unknownしてることを期待
    try:
        df_processed = predictor.processor.encode_categorical(df_processed, cat_cols)
    except Exception as e:
        print(f"Warning: Categorical encoding issue: {e}")
    
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
    results_df['odds'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(10.0)
    results_df['expected_value'] = results_df['probability'] * results_df['odds']
    
    # 5. ポートフォリオ配分 (Flex + Contrarian)
    recs_flex = BettingAllocator.allocate_budget(
        results_df, 
        budget=DEFAULT_BUDGET_PER_STRATEGY, 
        strategy='formation_flex',
        odds_data=odds_data
    )
    
    recs_contrarian = BettingAllocator.allocate_budget(
        results_df, 
        budget=DEFAULT_BUDGET_PER_STRATEGY, 
        strategy='meta_contrarian',
        odds_data=odds_data
    )
    
    race_info = {
        'race_id': race_id,
        'race_name': df_shutuba.iloc[0].get('レース名', 'Unknown Race'),
        'race_time': df_shutuba.attrs.get('race_data01', ''),
        'venue': get_venue_name(race_id)
    }
    
    return {
        'info': race_info,
        'portfolio': {
            'formation_flex': recs_flex,
            'meta_contrarian': recs_contrarian
        },
        'predictions': results_df.sort_values('probability', ascending=False).head(5)[['馬番', '馬名', 'probability', '単勝']].to_dict('records')
    }

def get_venue_name(race_id):
    place_code = race_id[4:6]
    codes = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    return codes.get(place_code, 'Unknown')

def load_historical_data():
    """過去データをロード（特徴量生成用）"""
    import pickle
    print("Loading historical data...")
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    horse_results = None
    peds = None
    if os.path.exists(hr_path):
        with open(hr_path, 'rb') as f: horse_results = pickle.load(f)
    if os.path.exists(peds_path):
        with open(peds_path, 'rb') as f: peds = pickle.load(f)
            
    return horse_results, peds

def main():
    parser = argparse.ArgumentParser(description="Predict races with Portfolio Strategy")
    parser.add_argument('--date', type=str, help='Target date YYYYMMDD', default=None)
    parser.add_argument('--budget', type=int, help='Budget per strategy', default=5000)
    args = parser.parse_args()
    
    global DEFAULT_BUDGET_PER_STRATEGY
    DEFAULT_BUDGET_PER_STRATEGY = args.budget

    date_str = get_target_date(args.date)
    print(f"Target Date: {date_str}")
    
    race_ids = get_race_ids(date_str)
    
    if not race_ids:
        print("No races found.")
        return

    
    # 1 Venue for Demo (Fast)
    if len(race_ids) > 12:
        print("Limiting to 12 races for demo speed...")
        race_ids = race_ids[:12]

    predictor = load_prediction_pipeline()
    horse_results, peds = load_historical_data()
    
    all_results = []
    
    for rid in tqdm(race_ids, desc="Processing Races"):
        try:
            res = process_race(rid, predictor, horse_results_db=horse_results, peds_db=peds)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\nError processing {rid}: {e}")
            
    # Output Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"portfolio_prediction_{date_str}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# ポートフォリオ予想レポート ({date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]})\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"戦略予算: 各 {DEFAULT_BUDGET_PER_STRATEGY}円 (計 {DEFAULT_BUDGET_PER_STRATEGY*2}円/レース)\n\n")
        
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
                
                f.write(f"### {venue}{r_num}R: {info['race_name']}\n")
                f.write(f"- 発走: {info['race_time']}\n\n")
                
                # Portfolio Table
                f.write("#### 🛡️ ポートフォリオ推奨買い目\n")
                
                # 1. Formation Flex
                recs_flex = race['portfolio']['formation_flex']
                if recs_flex:
                    f.write(f"**【攻め: Formation Flex】**\n")
                    for r in recs_flex:
                        r_type = r.get('bet_type', r.get('type', 'Unknown'))
                        r_method = r.get('method', 'Unknown')
                        r_desc = r.get('combination', r.get('description', r.get('desc', '')))
                        r_amount = r.get('total_amount', r.get('amount', 0))
                        f.write(f"- {r_type} {r_method}: {r_desc} ({r_amount}円)\n")
                else:
                    f.write("**【攻め: Formation Flex】** 推奨なし\n")
                    
                # 2. Meta Contrarian
                recs_cont = race['portfolio']['meta_contrarian']
                if recs_cont:
                    f.write(f"\n**【守り/一撃: Meta Contrarian】**\n")
                    for r in recs_cont:
                        r_type = r.get('bet_type', r.get('type', 'Unknown'))
                        r_method = r.get('method', 'Unknown')
                        r_desc = r.get('description', r.get('desc', ''))
                        r_amount = r.get('total_amount', r.get('amount', 0))
                        f.write(f"- {r_type} {r_method}: {r_desc} ({r_amount}円)\n")
                        # 買い目詳細
                        if 'horses' in r:
                             f.write(f"  - 対象: {r['horses']}\n")
                else:
                    f.write("\n**【守り/一撃: Meta Contrarian】** 推奨なし\n")
                
                f.write("\n#### 📊 AI注目馬 (Top 5)\n")
                f.write("| 番 | 馬名 | 勝率 | 単勝 |\n")
                f.write("|:-:|:---|:-:|:-:|\n")
                preds = race['predictions']
                for p in preds:
                    odds_str = f"{p['単勝']}倍" if p['単勝'] else "-"
                    f.write(f"| {p['馬番']} | {p['馬名']} | {p['probability']:.1%} | {odds_str} |\n")
                f.write("\n---\n")
                
    print(f"\nReport generated: {report_file}")

if __name__ == "__main__":
    main()
