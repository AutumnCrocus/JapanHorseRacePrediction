
import os
import sys
import re
import time
import pickle
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import RacePredictor
from modules.betting_allocator import BettingAllocator

class RankingWrapper:
    def __init__(self, data):
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = 'ltr'
    def predict(self, X):
        # LTRモデルのスコアを返す
        return self.model.predict(X[self.feature_names])
    def get_feature_importance(self, top_n=15):
        importances = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({'feature': self.feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(top_n)

def load_ltr_pipeline():
    """LTRモデルと前処理パイプラインをロード"""
    print("Loading LTR model...")
    
    ltr_model_path = os.path.join(MODEL_DIR, 'standalone_ranking', 'ranking_model.pkl')
    with open(ltr_model_path, 'rb') as f:
        data = pickle.load(f)
    
    model = RankingWrapper(data)
    
    # Processor / Engineer (LGBM版を流用)
    latest_model_dir = os.path.join(MODEL_DIR, 'historical_2010_2026')
    processor_path = os.path.join(latest_model_dir, 'processor.pkl')
    engineer_path = os.path.join(latest_model_dir, 'engineer.pkl')
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
        
    return RacePredictor(model, processor, engineer)

def load_historical_data():
    """過去データをロード"""
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

def process_race(race_id, predictor, budget=1000, horse_results_db=None, peds_db=None):
    """1レース分の予測実行"""
    print(f"  Predicting {race_id}...")
    
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        return None

    # 日付設定 (2026-02-14)
    df_shutuba['date'] = pd.to_datetime('2026-02-14')

    # オッズ取得
    odds_data = Odds.scrape(race_id)
    if odds_data and 'tan' in odds_data:
        for idx, row in df_shutuba.iterrows():
            try:
                umaban = int(row['馬番'])
                if umaban in odds_data['tan']:
                     df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
            except: pass

    # 前処理
    df_processed = predictor.processor.process_results(df_shutuba)
    
    # 体重補完
    if '体重' in df_processed.columns:
        mean_weight = df_processed['体重'].mean()
        if pd.isna(mean_weight): mean_weight = 470.0
        df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
    if '体重変化' in df_processed.columns:
        df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

    # 特徴量生成
    if horse_results_db is not None:
        df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results_db)
        df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results_db)
    
    df_processed, _ = predictor.engineer.add_jockey_features(df_processed)
    
    if peds_db is not None:
        df_processed = predictor.engineer.add_pedigree_features(df_processed, peds_db)
        
    df_processed = predictor.engineer.add_odds_features(df_processed)
    
    # エンコード & 予測
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df_processed.columns]
    df_processed = predictor.processor.encode_categorical(df_processed, cat_cols)
    
    feature_names = predictor.model.feature_names
    X = pd.DataFrame(index=df_processed.index)
    for col in feature_names:
        X[col] = df_processed[col] if col in df_processed.columns else 0
    
    numeric_X = X.select_dtypes(include=[np.number])
    X[numeric_X.columns] = numeric_X.fillna(numeric_X.median())
    X = X.fillna(0)
    
    if predictor.processor.scaler:
        X = predictor.processor.transform_scale(X)

    # LTRスコア
    scores = predictor.model.predict(X)
    
    # 結果整形
    results_df = df_shutuba.copy()
    results_df['probability'] = scores # LTRスコアをprobabilityとして扱う (BettingAllocatorでのソート用)
    results_df['horse_number'] = results_df['馬番'].astype(int)
    results_df['horse_name'] = results_df['馬名']
    results_df['odds'] = pd.to_numeric(results_df['単勝'], errors='coerce').fillna(0)
    
    # 予算配分 (sanrenpuku_1axis)
    recs = BettingAllocator.allocate_budget(
        results_df, 
        budget=budget, 
        odds_data=odds_data,
        strategy='sanrenpuku_1axis'
    )
    
    return {
        'info': {
            'race_id': race_id,
            'race_name': df_shutuba.iloc[0].get('レース名', 'Unknown'),
            'venue': get_venue_name(race_id)
        },
        'recommendations': recs,
        'predictions': results_df.sort_values('probability', ascending=False).head(5)
    }

def get_venue_name(race_id):
    codes = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    return codes.get(race_id[4:6], 'Unknown')

def main():
    race_ids = []
    # 東京 1-12R
    race_ids += [f"2026050105{r:02}" for r in range(1, 13)]
    # 京都 1-12R
    race_ids += [f"2026080205{r:02}" for r in range(1, 13)]
    # 小倉 1-12R
    race_ids += [f"2026100107{r:02}" for r in range(1, 13)]
    
    print(f"Target: {len(race_ids)} races for 2026/02/14")
    
    predictor = load_ltr_pipeline()
    horse_results, peds = load_historical_data()
    
    all_results = []
    for rid in tqdm(race_ids):
        try:
            res = process_race(rid, predictor, budget=1000, horse_results_db=horse_results, peds_db=peds)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"Error in {rid}: {e}")

    # レポート作成
    report_file = "reports/prediction_20260214_ltr.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 2026年2月14日 競馬予想レポート (LTRモデル)\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("モデル: LambdaMART (Ranking)\n")
        f.write("戦略: LTR推奨：3連複 軸流し (ranking_anchor)\n")
        f.write("予算: 1レース最大1000円\n\n")
        
        venue_groups = {}
        for r in all_results:
            v = r['info']['venue']
            if v not in venue_groups: venue_groups[v] = []
            venue_groups[v].append(r)
            
        for venue, races in venue_groups.items():
            f.write(f"## {venue}開催\n\n")
            for race in races:
                info = race['info']
                f.write(f"### {venue}{info['race_id'][-2:]}R: {info['race_name']}\n")
                
                f.write("#### 推奨買い目\n")
                if not race['recommendations']:
                    f.write("- 推奨なし\n")
                else:
                    for rec in race['recommendations']:
                        f.write(f"- **{rec['bet_type']} {rec['method']}**: {rec['combination']} ({rec['total_amount']}円)\n")
                        f.write(f"  - 理由: {rec['reason']}\n")
                
                f.write("\n#### 指数上位 (Top 5)\n")
                f.write("| 馬番 | 馬名 | LTRスコア | 単勝オッズ |\n")
                f.write("| :---: | :--- | :---: | :---: |\n")
                for _, p in race['predictions'].iterrows():
                    f.write(f"| {p['horse_number']} | {p['horse_name']} | {p['probability']:.4f} | {p['odds']}倍 |\n")
                f.write("\n---\n")
                
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    main()
