"""
2026-02-14 予想スクリプト
- モデル: LightGBM (EnsembleModel)
- 戦略: box4_sanrenpuku (3連複4頭BOX)
- 予算: 各レース1,000円
"""

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

# --- Configuration ---
DATE_STR = '20260214'
STRATEGY = 'box4_sanrenpuku'
BUDGET = 1000
REPORT_DIR = 'reports'


def get_race_ids(date_str: str) -> list[str]:
    """
    指定された日付のレースIDを取得する。
    URL: https://race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD
    """
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    print(f"Fetching race IDs from: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
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
            print("No race IDs found via list page. Using brute-force scan...")
            race_ids = scan_race_ids_brute_force(date_str)
            
        return race_ids
    except Exception as e:
        print(f"Error fetching race IDs: {e}")
        return []


def scan_race_ids_brute_force(date_str: str) -> list[str]:
    """
    総当たりでその日に開催されるレースIDを特定する。
    ID形式: YYYY PP KK DD RR
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    year = date_str[:4]
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    target_date_jp = f"{year}年{month}月{day}日"
    print(f"Target Date: {target_date_jp}")
    
    places = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    kais = range(1, 6)
    days = range(1, 13)
    
    keys = []
    for p in places:
        for k in kais:
            for d in days:
                keys.append(f"{year}{p:02}{k:02}{d:02}")
    
    active_keys = []
    
    def check_key(key: str):
        rid = key + "01"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.encoding = 'EUC-JP'
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    print(f"  Found: {rid}")
                    return key
            return None
        except:
            return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in tqdm(as_completed(futures), total=len(keys), desc="Searching"):
            result = future.result()
            if result:
                active_keys.append(result)
    
    print(f"Active keys: {active_keys}")
    
    final_ids = []
    for key in sorted(active_keys):
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
            
    return final_ids


def load_prediction_pipeline() -> RacePredictor:
    """モデルと前処理パイプラインをロードする。"""
    import pickle
    print("Loading models...")
    
    if os.path.exists(os.path.join(MODEL_DIR, 'model_lgbm_0.pkl')):
        model = EnsembleModel()
        model.load(MODEL_DIR)
        print("Ensemble (LGBM) model loaded.")
    else:
        model = HorseRaceModel()
        model.load()
        print("Single model loaded.")

    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)


def load_historical_data():
    """過去データをロードする（特徴量生成用）。"""
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
            
        print("Pre-calculating pedigree scores...")
        try:
            ped_scores = {
                'speed': ['ディープインパクト', 'ロードカナロア', 'サクラバクシンオー', 'ダイワメジャー', 'キングカメハメハ'],
                'stamina': ['ハーツクライ', 'オルフェーヴル', 'ゴールドシップ', 'ステイゴールド', 'エピファネイア'],
                'dirt': ['ヘニーヒューズ', 'シニスターミニスター', 'ゴールドアリュール', 'パイロ', 'クロフネ']
            }
            
            peds_str_series = peds.fillna('').astype(str).agg(' '.join, axis=1)
            
            for cat, sires in ped_scores.items():
                def count_s(text, sires=sires):
                    c = 0
                    for s in sires:
                        if s in text: c += 1
                    return c
                peds[f'peds_score_{cat}'] = peds_str_series.apply(lambda x: count_s(x))
            print("Pedigree scores calculated.")
        except Exception as e:
            print(f"Error calculating pedigree scores: {e}")
            
    return horse_results, peds


def get_venue_name(race_id: str) -> str:
    """レースIDから会場名を取得する。"""
    place_code = race_id[4:6]
    codes = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    return codes.get(place_code, 'Unknown')


def process_race(race_id: str, predictor: RacePredictor, budget: int = 1000, 
                 strategy: str = 'box4_sanrenpuku',
                 horse_results_db=None, peds_db=None) -> dict | None:
    """1レース分の処理を実行する。"""
    print(f"\nProcessing Race ID: {race_id}")
    
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        print("  Failed to fetch shutuba data.")
        return None

    # リスト値のクリーニング
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)

    # 日付カラムを追加
    race_date = f"{DATE_STR[:4]}-{DATE_STR[4:6]}-{DATE_STR[6:8]}"
    df_shutuba['date'] = pd.to_datetime(race_date)

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
        filled_count = df_processed['体重'].isna().sum()
        if filled_count > 0:
            df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    # 履歴特徴量
    if horse_results_db is not None:
         df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
         df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed)
    
    if peds_db is not None:
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
    
    # 5. 予算配分 (box4_sanrenpuku 戦略)
    recommendations = BettingAllocator.allocate_budget(
        results_df, 
        budget=budget, 
        odds_data=odds_data,
        strategy=strategy
    )
    
    race_info = {
        'race_id': race_id,
        'race_name': df_shutuba.iloc[0].get('レース名', 'Unknown Race'),
        'race_time': df_shutuba.attrs.get('race_data01', ''),
        'venue': get_venue_name(race_id)
    }
    
    return {
        'info': race_info,
        'recommendations': recommendations,
        'predictions': results_df.sort_values('probability', ascending=False).head(5)[['馬番', '馬名', 'probability', '単勝']].to_dict('records')
    }


def generate_report(all_results: list, date_str: str, strategy: str, budget: int) -> str:
    """予想レポートを生成する。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_file = os.path.join(REPORT_DIR, f"prediction_{date_str}_{strategy}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬予想レポート ({date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]})\n\n")
        f.write(f"- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- モデル: LGBM (Ensemble)\n")
        f.write(f"- 戦略: {strategy}\n")
        f.write(f"- 各レース予算: {budget}円\n")
        f.write("- ※馬体重は未発表のため、平均値等で補完して予測しています。\n\n")
        
        # 会場ごとにグループ化
        venue_groups = {}
        for r in all_results:
            v = r['info']['venue']
            if v not in venue_groups: venue_groups[v] = []
            venue_groups[v].append(r)
            
        total_bet_amount = 0
        
        for venue, races in venue_groups.items():
            f.write(f"## {venue}開催\n\n")
            for race in races:
                info = race['info']
                rid = info['race_id']
                r_num = rid[-2:]
                
                f.write(f"### {venue}{int(r_num)}R: {info['race_name']}\n")
                f.write(f"- 発走情報: {info['race_time']}\n\n")
                
                f.write(f"#### 推奨買い目 (予算{budget}円)\n")
                recs = race['recommendations']
                if not recs:
                    f.write("- 推奨なし (オッズ不足または混戦)\n")
                else:
                    for rec in recs:
                        f.write(f"- **{rec['bet_type']} {rec['method']}**: {rec['combination']} ({rec['total_amount']}円)\n")
                        f.write(f"  - 理由: {rec['reason']}\n")
                        total_bet_amount += rec['total_amount']
                
                f.write(f"\n#### AI注目馬 (Top 5)\n")
                f.write("| 馬番 | 馬名 | 勝率予測 | 単勝オッズ |\n")
                f.write("| :---: | :--- | :---: | :---: |\n")
                preds = race['predictions']
                for p in preds:
                    odds_str = f"{p['単勝']}倍" if p['単勝'] else "-"
                    f.write(f"| {p['馬番']} | {p['馬名']} | {p['probability']:.1%} | {odds_str} |\n")
                f.write("\n---\n\n")
        
        f.write(f"\n## サマリー\n\n")
        f.write(f"- 処理レース数: {len(all_results)}\n")
        f.write(f"- 合計投資額: {total_bet_amount}円\n")
                
    return report_file


def main():
    """メイン処理。"""
    print(f"=== 競馬予想 ({DATE_STR}) ===")
    print(f"戦略: {STRATEGY} / 予算: {BUDGET}円/レース")
    print()
    
    race_ids = get_race_ids(DATE_STR)
    
    if not race_ids:
        print("No races found.")
        return

    predictor = load_prediction_pipeline()
    horse_results, peds = load_historical_data()
    
    all_results = []
    
    for rid in tqdm(race_ids, desc="Processing Races"):
        try:
            res = process_race(
                rid, predictor, 
                budget=BUDGET, 
                strategy=STRATEGY,
                horse_results_db=horse_results, 
                peds_db=peds
            )
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\nError processing {rid}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        report_file = generate_report(all_results, DATE_STR, STRATEGY, BUDGET)
        print(f"\nReport generated: {report_file}")
    else:
        print("\nNo results to report.")


if __name__ == "__main__":
    main()
