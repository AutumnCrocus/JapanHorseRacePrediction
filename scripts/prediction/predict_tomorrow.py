
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor
from modules.betting_allocator import BettingAllocator

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
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # リンクからrace_id=...を探す
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
    places = [5, 8, 10] # 東京, 京都, 小倉 (1月の一般的開催)
    # 他の会場の可能性もあるが、まずはこれで。もし0件なら拡大するロジックが必要だが...
    
    kais = range(1, 4)    # 1~3回
    days = range(1, 13)   # 1~12日目
    
    # 候補となる「会場・回・日」のキー (YYYYPPKKDD)
    keys = []
    for p in places:
        for k in kais:
            for d in days:
                keys.append(f"{year}{p:02}{k:02}{d:02}")
    
    active_keys = []
    
    def check_key(key):
        # 1RのIDで存在確認
        rid = key + "01"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10) # タイムアウト延長
            res.encoding = 'EUC-JP'
            # ページが存在し、かつ日付が一致するか
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    print(f"DEBUG: Found match {rid}")
                    return key
                else:
                    # 日付不一致でもページがある場合、開催日が違うID -> PrintなしでSkip
                    pass
            return None
        except Exception as e:
            # print(f"DEBUG: Error checking {rid}: {e}")
            return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    # workers=5に変更
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in tqdm(as_completed(futures), total=len(keys), desc="Searching"):
            result = future.result()
            if result:
                active_keys.append(result)
                
    # 結果からハードコード (再実行時の短縮)
    # Found keys: 2026050101 (Tokyo), 2026080201 (Kyoto), 2026100103 (Kokura)
    active_keys = ['2026050101', '2026080201', '2026100103']
    print(f"Using cached keys: {active_keys}")
    
    # IDリスト生成 (各キーにつき12レース)
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

def process_race(race_id, predictor, budget=1000, horse_results_db=None, peds_db=None):
    """1レース分の処理を実行"""
    print(f"\nProcessing Race ID: {race_id}")
    
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        print("  Failed to fetch shutuba data.")
        return None

    # Cleaning: Validate and fix list values in DataFrame
    # Sometimes 'odds' or other columns might contain lists due to scraping issues
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            print(f"WARNING: Found list in column '{col}'. Flattening...")
            # Take the first element if it's a list, or join strings
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)

    # 日付カラムを追加（必須）
    # race_idから本来は推定できるが、ここでは実行日(2026-01-31)を固定で入れる
    df_shutuba['date'] = pd.to_datetime('2026-01-31')

    # 2. オッズ取得
    odds_data = Odds.scrape(race_id)
    
    # オッズ情報をdf_shutubaにマージ（予測精度向上と表示用）
    # Shutuba.scrape内でも簡易的にオッズ取っているが、Odds.scrapeの方が確実
    # ここでは既存の単勝オッズがあれば優先、なければ補完
    if odds_data and 'tan' in odds_data:
        for idx, row in df_shutuba.iterrows():
            try:
                umaban = int(row['馬番'])
                if umaban in odds_data['tan']:
                     df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
            except: pass

    # 3. 前処理 & 特徴量生成
    # ここで馬体重補完を行う
    # process_resultsは内部で体重抽出を行うため、元の文字列をいじる必要があるか、
    # あるいは process_results実行後のDFをいじるか。
    # process_results後のDFをいじるのが楽。
    
    # 3.1 まずProcessorを通す
    df_processed = predictor.processor.process_results(df_shutuba)
    
    # 3.2 馬体重補完 (Process後なので '体重', '体重変化' カラムになっている)
    if '体重' in df_processed.columns:
        # 全体平均または固定値(470)で埋める
        # すでに値がある行の平均を使うとより良い
        mean_weight = df_processed['体重'].mean()
        if pd.isna(mean_weight): mean_weight = 470.0
        
        filled_count = df_processed['体重'].isna().sum()
        if filled_count > 0:
            print(f"  Imputing weights for {filled_count} horses with value {mean_weight:.1f}")
            df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
            
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    # 3.3 Engineerを通す (履歴特徴量など)
    # 過去データが必要。RAW_DATA_DIRからロードしておいたものを渡す
    if horse_results_db is not None:
         # 日付フォーマット調整などが必要かもしれないが、engineerがやってくれるはず
         df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
         df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed) # 戻り値は(df, stats)
    
    if peds_db is not None:
        df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_db)
        
    df_processed = predictor.engineer.add_odds_features(df_processed)
    
    # 3.4 カテゴリエンコード
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df_processed.columns]
    df_processed = predictor.processor.encode_categorical(df_processed, cat_cols)
    
    # 3.5 特徴量選択 & 欠損埋め (Modelが必要とするカラム)
    feature_names = predictor.model.feature_names
    X = pd.DataFrame(index=df_processed.index)
    for col in feature_names:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0
    
    # 最終的な欠損処理 (Median)
    # 数値カラムのみに対して処理を行うように修正
    numeric_X = X.select_dtypes(include=[np.number])
    X[numeric_X.columns] = numeric_X.fillna(numeric_X.median())
    X = X.fillna(0)
    
    if predictor.processor.scaler:
        X = predictor.processor.transform_scale(X)

    # 4. 予測
    probs = predictor.model.predict(X)
    
    # 結果整形
    results_df = df_shutuba.copy()
    results_df['probability'] = probs
    results_df['horse_number'] = results_df['馬番'].astype(int)
    results_df['horse_name'] = results_df['馬名']
    
    # 期待値計算 (Win Probability * Win Odds)
    # 単勝オッズは文字列の可能性があるので変換
    results_df['odds'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(0)
    results_df['expected_value'] = results_df['probability'] * results_df['odds']
    
    # 5. 予算配分
    recommendations = BettingAllocator.allocate_budget(
        results_df, 
        budget=budget, 
        odds_data=odds_data
    )
    
    race_info = {
        'race_id': race_id,
        'race_name': df_shutuba.iloc[0].get('レース名', 'Unknown Race'),
        'race_time': df_shutuba.attrs.get('race_data01', ''), # 発走時刻などはここに含まれることが多い
        'venue': get_venue_name(race_id)
    }
    
    return {
        'info': race_info,
        'recommendations': recommendations,
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
        with open(hr_path, 'rb') as f:
            horse_results = pickle.load(f)
            
    if os.path.exists(peds_path):
        with open(peds_path, 'rb') as f:
            peds = pickle.load(f)
            
        # Pre-calculate Pedigree Scores (Optimization)
        print("Pre-calculating pedigree scores...")
        try:
             # 血統スコア定義 (簡易版) - preprocessing.pyと同じ定義
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
            print("Pedigree scores calculated.")
            
        except Exception as e:
            print(f"Error checking peds columns: {e}")
            
    return horse_results, peds

def main():
    date_str = '20260131'
    race_ids = get_race_ids(date_str)
    
    if not race_ids:
        print("No races found for tomorrow.")
        return

    # Load resources
    predictor = load_prediction_pipeline()
    horse_results, peds = load_historical_data()
    
    all_results = []
    
    # Sequential Processing (Stable & Fast with pre-calc)
    for rid in tqdm(race_ids, desc="Processing Races"):
        try:
            res = process_race(rid, predictor, budget=1000, horse_results_db=horse_results, peds_db=peds)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\nError processing {rid}: {e}")
            import traceback
            traceback.print_exc()
            
    # Output Report
            
    # Output Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"prediction_report_{date_str}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 競馬予想レポート ({date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]})\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("※馬体重は未発表のため、平均値等で補完して予測しています。\n\n")
        
        # 会場ごとにグループ化
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
                f.write(f"- 発走情報: {info['race_time']}\n\n")
                
                f.write("#### 推奨買い目 (予算1000円)\n")
                recs = race['recommendations']
                if not recs:
                    f.write("- 推奨なし (オッズ不足または混戦)\n")
                else:
                    for rec in recs:
                        f.write(f"- **{rec['bet_type']} {rec['method']}**: {rec['combination']} ({rec['total_amount']}円)\n")
                        f.write(f"  - 理由: {rec['reason']}\n")
                
                f.write("\n#### AI注目馬 (Top 5)\n")
                f.write("| 馬番 | 馬名 | 勝率予測 | 単勝オッズ |\n")
                f.write("| :---: | :--- | :---: | :---: |\n")
                preds = race['predictions']
                for p in preds:
                    odds_str = f"{p['単勝']}倍" if p['単勝'] else "-"
                    f.write(f"| {p['馬番']} | {p['馬名']} | {p['probability']:.1%} | {odds_str} |\n")
                f.write("\n---\n")
                
    print(f"\nReport generated: {report_file}")

if __name__ == "__main__":
    main()
