"""
軽量動作確認テスト: 予測パイプライン
1レースだけの出馬表を使って、前処理→予測までがエラーなく通るかを確認する。
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

# モジュールパス
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.scraping import Shutuba
from modules.preprocessing import FeatureEngineer
from modules.training import HorseRaceModel
from modules.constants import MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE, RACE_TYPE_MAP

def test_prediction_pipeline():
    print("=== 高速動作確認テスト ===")
    
    # 1. 1レースだけ取得 (中山1R)
    rid = "202606010901" 
    print(f"Fetching {rid}...")
    try:
        df = Shutuba.scrape(rid)
    except Exception as e:
        print(f"Scraping failed: {e}")
        # 失敗時はダミーデータ作成
        df = pd.DataFrame({
            '枠番': [1, 2, 3], '馬番': [1, 2, 3], '馬名': ['テストA', 'テストB', 'テストC'],
            '性齢': ['牡3', '牝3', 'セ4'], '斤量': [56, 54, 56],
            'コース': ['芝2000m'] * 3, '単勝': [2.5, 5.0, 10.0], '人気': [1, 2, 3],
            '騎手': ['騎手A', '騎手B', '騎手C'], '調教師': ['調教師A', '調教師B', '調教師C'],
            '馬体重': ['480(0)', '460(+2)', '500(-2)'],
            'jockey_id': ['j1', 'j2', 'j3'], 'trainer_id': ['t1', 't2', 't3'],
            'horse_id': ['h1', 'h2', 'h3'],
            'sire': ['父A', '父B', '父C'], 'dam': ['母A', '母B', '母C']
        }, index=[rid]*3)
    
    if df.empty:
        print("Empty dataframe.")
        return

    print("Data loaded.")
    
    # 2. モデルロード
    model = HorseRaceModel()
    model.load(os.path.join(MODEL_DIR, 'horse_race_model.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    
    # ファイルがあればロード、なければNone
    try: bias_map = pd.read_pickle(os.path.join(MODEL_DIR, 'bias_map.pkl'))
    except: bias_map = None
    try: jockey_stats = pd.read_pickle(os.path.join(MODEL_DIR, 'jockey_stats.pkl'))
    except: jockey_stats = None
    try: hr_df = pd.read_pickle(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
    except: hr_df = pd.DataFrame()
    
    # 3. 前処理 (predict_tomorrowと同じロジック)
    target_date = "2026/01/25"
    df['date'] = pd.to_datetime(target_date)
    
    # ID生成
    rid_str = df.index.astype(str)
    df['venue_id'] = pd.to_numeric(rid_str.str[4:6], errors='coerce').fillna(0).astype(int)
    df['kai'] = pd.to_numeric(rid_str.str[6:8], errors='coerce').fillna(0).astype(int)
    df['day'] = pd.to_numeric(rid_str.str[8:10], errors='coerce').fillna(0).astype(int)
    df['race_num'] = pd.to_numeric(rid_str.str[10:12], errors='coerce').fillna(0).astype(int)
    
    # コース
    if 'コース' in df.columns:
        extracted = df['コース'].astype(str).str.extract(r'([芝ダ障])(\d+)')
        df['race_type_str'] = extracted[0]
        df['course_len'] = pd.to_numeric(extracted[1], errors='coerce').fillna(2000).astype(int)
        df['race_type'] = df['race_type_str'].map(RACE_TYPE_MAP).fillna(0).astype(int)
    else:
        df['course_len'] = 2000; df['race_type'] = 0

    # 数値化
    df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(0).astype(int)
    df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(0).astype(int)
    df['斤量'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(56.0)
    
    if '性齢' in df.columns:
        sex_map = {'牡': 0, '牝': 1, 'セ': 2}
        df['性'] = df['性齢'].str[0].map(sex_map).fillna(0).astype(int)
        df['年齢'] = pd.to_numeric(df['性齢'].str[1:], errors='coerce').fillna(4).astype(int)

    if '単勝' in df.columns: df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(10.0)
    if '人気' in df.columns: df['人気'] = pd.to_numeric(df['人気'], errors='coerce').fillna(5)

    # Features
    if not hr_df.empty:
        hr_df.columns = hr_df.columns.str.replace(' ', '')
        if '着順' in hr_df.columns: hr_df['着順'] = pd.to_numeric(hr_df['着順'], errors='coerce')
        df = engineer.add_horse_history_features(df, hr_df)
        df = engineer.add_course_suitability_features(df, hr_df)
        
    df, _ = engineer.add_jockey_features(df, jockey_stats=jockey_stats)
    
    if bias_map is not None:
        df = engineer.add_bias_features(df, bias_map)
    else:
        df['waku_bias_rate'] = 0.3
        
    # Encode
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = processor.encode_categorical(df, cat_cols)
    
    # Prepare X
    for c in model.feature_names:
        if c not in df.columns: df[c] = 0
    
    # 先にfillna(0) (数値型のまま処理)
    X = df[model.feature_names].fillna(0)
    
    # その後、学習時と同様にカテゴリ型へ変換
    for col in ['枠番', '馬番']:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    print(f"Features ready. Shape: {X.shape}")
    print(f"Dtypes:\n{X.dtypes}")

    # Predict
    print("Predicting...")
    probs = model.predict(X)
    print("Predictions:", probs)
    print("Test PASSED.")

if __name__ == '__main__':
    test_prediction_pipeline()
