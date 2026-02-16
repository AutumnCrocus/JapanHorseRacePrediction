                                                                                                                                                                                    
import os
import sys
import pickle
import requests
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import (
    MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE,
    HORSE_RESULTS_FILE, PEDS_FILE, HEADERS
)
from modules.scraping import Shutuba, Odds

# ============= 設定 =============
TARGET_DATE = '20260215'
LTR_MODEL_DIR = os.path.join(MODEL_DIR, "historical_ltr_2010_2024")
LGBM_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
OUTPUT_FILE = f"data/processed/prediction_{TARGET_DATE}.csv"
# ===============================

class RankingWrapper:
    """LTRモデルのラッパー"""
    def __init__(self, data: dict):
        self.model = data['model']
        self.feature_names = data['feature_names']

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])



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
    
    # 探索範囲 (東京05, 京都08, 小倉10)
    places = [5, 8, 10]
    kais = range(1, 4)    # 1~3回
    days = range(1, 13)   # 1~12日目
    
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
            res = requests.get(url, headers=HEADERS, timeout=5)
            res.encoding = 'EUC-JP'
            if res.status_code == 200 and "出馬表" in res.text:
                if target_date_jp in res.text:
                    print(f"DEBUG: Found match {rid}")
                    return key
            return None
        except:
            return None

    print(f"Scanning {len(keys)} potential venue/dates...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_key, k) for k in keys]
        for future in as_completed(futures):
            result = future.result()
            if result:
                active_keys.append(result)
                
    final_ids = []
    for key in sorted(active_keys):
        for r in range(1, 13):
            final_ids.append(f"{key}{r:02}")
            
    print(f"Brute-force scan found {len(final_ids)} races.")
    return final_ids

def get_race_ids(date_str):
    """指定された日付のレースIDを取得する"""
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
        print(f"Found {len(race_ids)} races via list page.")
        
        if not race_ids:
            print("Trying brute-force scan...")
            race_ids = scan_race_ids_brute_force(date_str)
            
        return race_ids
    except Exception as e:
        print(f"Error fetching race IDs: {e}")
        return []

def load_models() -> dict:
    """モデルとプロセッサをロード"""
    models: dict = {}
    
    # 1. LTR Model
    if os.path.exists(LTR_MODEL_DIR):
        print("Loading LTR model...")
        with open(os.path.join(LTR_MODEL_DIR, 'ranking_model.pkl'), 'rb') as f:
            data = pickle.load(f)
        models['ltr'] = {
            'model': RankingWrapper(data),
            'processor': pickle.load(open(os.path.join(LTR_MODEL_DIR, 'processor.pkl'), 'rb')),
            'engineer': pickle.load(open(os.path.join(LTR_MODEL_DIR, 'engineer.pkl'), 'rb')),
        }
    
    # 2. LGBM Model (Box4用)
    if os.path.exists(LGBM_MODEL_DIR):
        print("Loading LGBM model...")
        lgbm_model_path = os.path.join(LGBM_MODEL_DIR, "model.pkl")
        with open(lgbm_model_path, 'rb') as f:
            lgbm_data = pickle.load(f)
        
        if isinstance(lgbm_data, dict):
            booster = lgbm_data['model']
            feat_names = lgbm_data.get('feature_names', [])
        else:
            booster = lgbm_data
            feat_names = booster.feature_name()
            
        models['lgbm'] = {
            'booster': booster,
            'feature_names': feat_names,
            'processor': pickle.load(open(os.path.join(LGBM_MODEL_DIR, 'processor.pkl'), 'rb')),
            'engineer': pickle.load(open(os.path.join(LGBM_MODEL_DIR, 'engineer.pkl'), 'rb')),
        }
    return models

def load_historical_data():
    """過去データをロード"""
    print("Loading historical data...")
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    return hr, peds

def process_single_race(race_id, models, hr, peds, date_val):
    """1レース分の処理"""
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty: return pd.DataFrame()
    
    # 日付設定
    df_shutuba['date'] = date_val
    
    # 2. オッズ取得 (特徴量用)
    odds_data = Odds.scrape(race_id)
    # 単勝オッズ補完
    if odds_data and 'tan' in odds_data:
        for idx, row in df_shutuba.iterrows():
            try:
                umaban = int(row['馬番'])
                if umaban in odds_data['tan']:
                     df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
            except: pass

    # 結果格納用DataFrame作成
    result_df = df_shutuba[['馬番', '馬名']].copy()
    result_df['race_id'] = race_id
    result_df['date'] = date_val
    result_df['horse_number'] = result_df['馬番'].astype(int)
    result_df['horse_name'] = result_df['馬名']
    result_df['win_odds'] = pd.to_numeric(df_shutuba['単勝'], errors='coerce').fillna(0)
    
    # モデルごとの予測処理
    for model_name, env in models.items():
        try:
            # 前処理 (Deep Copyして副作用回避)
            df_proc = env['processor'].process_results(df_shutuba.copy())
            
            # 特徴量生成
            df_proc = env['engineer'].add_horse_history_features(df_proc, hr)
            df_proc = env['engineer'].add_course_suitability_features(df_proc, hr)
            df_proc, _ = env['engineer'].add_jockey_features(df_proc)
            df_proc = env['engineer'].add_pedigree_features(df_proc, peds)
            df_proc = env['engineer'].add_odds_features(df_proc)
            
            # カテゴリエンコード
            cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
            cat_cols = [c for c in cat_cols if c in df_proc.columns]
            df_proc = env['processor'].encode_categorical(df_proc, cat_cols)

            # 特徴量抽出
            feature_names = env['model'].feature_names if model_name == 'ltr' else env['feature_names']
            X = pd.DataFrame(index=df_proc.index)
            for col in feature_names:
                X[col] = df_proc[col] if col in df_proc.columns else 0
            
            # 欠損埋め
            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
            X = X.fillna(0)
            
            # Object型対策
            if model_name == 'lgbm':
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category')

            # 予測
            score_col = 'ltr_score' if model_name == 'ltr' else 'prob_score'
            
            if model_name == 'ltr':
                scores = env['model'].predict(X)
                df_proc[score_col] = scores
            elif model_name == 'lgbm':
                probs = env['booster'].predict(X)
                df_proc[score_col] = probs
            
            # マージ (馬番キー)
            # df_procに馬番があるか確認
            if '馬番' in df_proc.columns:
                # 型合わせ
                df_proc['馬番'] = df_proc['馬番'].astype(int)
                
                # スコア抽出
                scores_df = df_proc[['馬番', score_col]].copy()
                
                # merge
                result_df = pd.merge(result_df, scores_df, left_on='horse_number', right_on='馬番', how='left')
                result_df.drop(columns=['馬番_y'], inplace=True, errors='ignore')
                result_df.rename(columns={'馬番_x': '馬番'}, inplace=True)
            else:
                print(f"Warning: '馬番' column lost in processing for {model_name}")

        except Exception as e:
            print(f"Error in {model_name} prediction for {race_id}: {e}")
            import traceback
            traceback.print_exc()

    return result_df

def main():
    print(f"Start prediction for {TARGET_DATE}...")
    
    # 1. レースID取得
    race_ids = get_race_ids(TARGET_DATE)
    if not race_ids:
        print("No races found.")
        return

    # 2. モデル & データロード
    models = load_models()
    hr, peds = load_historical_data()
    
    target_date_obj = pd.to_datetime(TARGET_DATE)

    # 3. 実行
    all_preds_list = []
    
    for rid in tqdm(race_ids, desc="Predicting races"):
        df = process_single_race(rid, models, hr, peds, target_date_obj)
        if not df.empty:
            all_preds_list.append(df)
            
    if not all_preds_list:
        print("No predictions generated.")
        return
        
    final_df = pd.concat(all_preds_list, ignore_index=True)
    
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to {OUTPUT_FILE} ({len(final_df)} rows)")

if __name__ == "__main__":
    main()
