"""
2026/01/25 レース予測スクリプト (修正完了版)
"""
import sys
import os
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle

# モジュールパスの追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.scraping import Shutuba as ShutubaTable
from modules.preprocessing import FeatureEngineer, prepare_training_data
from modules.training import HorseRaceModel
from modules.constants import MODEL_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE, RACE_TYPE_MAP, WEATHER_MAP, GROUND_MAP

def predict_tomorrow():
    print("=== 2026/01/25 レース予測 (Final) ===")
    
    # 開催日と場所コード
    target_date = "2026/01/25"
    kaisai_list = [
        {'id_base': '2026060109', 'name': '中山'},
        {'id_base': '2026080109', 'name': '京都'},
        {'id_base': '2026100102', 'name': '小倉'}
    ]
    
    # レースID生成 (全36レース)
    race_ids = []
    for kaisai in kaisai_list:
        for r in range(1, 13):
            race_ids.append(f"{kaisai['id_base']}{r:02d}")
            
    print(f"対象レース数: {len(race_ids)}")
    
    # 出馬表スクレイピング
    print("\n[1/4] 出馬表の取得...")
    shutuba_list = []
    
    def fetch_shutuba(rid):
        time.sleep(1) # サーバー負荷軽減
        return ShutubaTable.scrape(rid)

    # 並列実行
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_rid = {executor.submit(fetch_shutuba, rid): rid for rid in race_ids}
        for future in tqdm(as_completed(future_to_rid), total=len(race_ids), desc="出馬表取得"):
            df = future.result()
            if not df.empty:
                shutuba_list.append(df)
            
    if not shutuba_list:
        print("出馬表が取得できませんでした。")
        return

    shutuba_df = pd.concat(shutuba_list)
    print(f"取得データ数: {len(shutuba_df)}頭")
    
    # モデルとデータの読み込み
    print("\n[2/4] モデルと過去データの読み込み...")
    
    model_path = os.path.join(MODEL_DIR, 'horse_race_model.pkl')
    if not os.path.exists(model_path):
        print("学習済みモデルが見つかりません。")
        return
        
    model = HorseRaceModel()
    model.load(model_path)
    
    # Processor/Engineer読み込み
    with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    try:
        bias_map = pd.read_pickle(os.path.join(MODEL_DIR, 'bias_map.pkl'))
        print("Bias map loaded.")
    except:
        bias_map = None
    try:
        jockey_stats = pd.read_pickle(os.path.join(MODEL_DIR, 'jockey_stats.pkl'))
        print("Jockey stats loaded.")
    except:
        jockey_stats = None
        
    # 過去成績と血統データの読み込み
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else pd.DataFrame()
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else pd.DataFrame()
    
    # 前処理
    print("\n[3/4] データ前処理...")
    df = shutuba_df.copy()
    
    # 1. 日付設定
    df['date'] = pd.to_datetime(target_date)
    
    # 2. ID系カラム生成
    rid_str = df.index.astype(str)
    df['venue_id'] = pd.to_numeric(rid_str.str[4:6], errors='coerce').fillna(0).astype(int)
    df['kai'] = pd.to_numeric(rid_str.str[6:8], errors='coerce').fillna(0).astype(int)
    df['day'] = pd.to_numeric(rid_str.str[8:10], errors='coerce').fillna(0).astype(int)
    df['race_num'] = pd.to_numeric(rid_str.str[10:12], errors='coerce').fillna(0).astype(int)
    
    # 3. コース情報の抽出 (距離、タイプ)
    if 'コース' in df.columns:
        extracted = df['コース'].astype(str).str.extract(r'([芝ダ障])(\d+)')
        df['race_type_str'] = extracted[0]
        df['course_len'] = pd.to_numeric(extracted[1], errors='coerce').fillna(2000).astype(int)
        
        # マッピング
        df['race_type'] = df['race_type_str'].map(RACE_TYPE_MAP).fillna(0).astype(int)
    else:
        df['course_len'] = 2000
        df['race_type'] = 0

    # 4. 数値化処理
    df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(0).astype(int)
    df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(0).astype(int)
    df['斤量'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(56.0)
    
    if '性齢' in df.columns:
        sex_map = {'牡': 0, '牝': 1, 'セ': 2}
        df['性'] = df['性齢'].str[0].map(sex_map).fillna(0).astype(int)
        df['年齢'] = pd.to_numeric(df['性齢'].str[1:], errors='coerce').fillna(4).astype(int)
    else:
        df['性'] = 0
        df['年齢'] = 4
    
    if '単勝' in df.columns:
        df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(10.0)
    if '人気' in df.columns:
        df['人気'] = pd.to_numeric(df['人気'], errors='coerce').fillna(5)

    # 5. 特徴量エンジニアリング
    if not hr_df.empty:
        hr_df.columns = hr_df.columns.str.replace(' ', '')
        if '着順' in hr_df.columns:
            hr_df['着順'] = pd.to_numeric(hr_df['着順'], errors='coerce')
        df = engineer.add_horse_history_features(df, hr_df)
        df = engineer.add_course_suitability_features(df, hr_df)
    
    if not peds_df.empty:
        df = engineer.add_pedigree_features(df, peds_df)
        
    df, _ = engineer.add_jockey_features(df, jockey_stats=jockey_stats)
    
    if bias_map is not None:
        df = engineer.add_bias_features(df, bias_map)
    else:
        df['waku_bias_rate'] = 0.3
        
    # オッズ・人気特徴量 (NEW)
    df = engineer.add_odds_features(df)
    
    # 6. エンコード
    cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = processor.encode_categorical(df, cat_cols)
    
    features = [c for c in model.feature_names if c in df.columns]
    for c in model.feature_names:
        if c not in df.columns: df[c] = 0
        
    # Prepare X
    # 先にfillna(0) (数値型のまま処理)
    X = df[model.feature_names].fillna(0)
    
    # その後、学習時と同様にカテゴリ型へ変換 (枠番、馬番)
    for col in ['枠番', '馬番']:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    # 予測
    print("\n[4/4] 予測実行...")
    probs = model.predict(X)
    df['score'] = probs
    
    # 結果出力
    output_lines = []
    output_lines.append(f"# 📅 {target_date} 厳選・推奨買い目リスト")
    
    race_strategies = []
    for race_id in sorted(df.index.unique()):
        race_df = df[df.index == race_id].copy().sort_values('score', ascending=False)
        race_name = race_df.iloc[0].get('レース名', 'Unknown Race')
        top1 = race_df.iloc[0]
        top2 = race_df.iloc[1] if len(race_df) > 1 else top1
        top3 = race_df.iloc[2] if len(race_df) > 2 else top2
        others = race_df.iloc[3:6] if len(race_df) > 3 else pd.DataFrame()
        
        score_diff = top1['score'] - top2['score']
        max_score = top1['score']
        
        if max_score >= 0.40 and score_diff >= 0.10:
            confidence = 'S' # 鉄板
            strategy_type = 'winner'
        elif max_score >= 0.35:
            confidence = 'A' # 有力
            strategy_type = 'standard'
        elif max_score >= 0.28:
            confidence = 'B' # 推奨
            strategy_type = 'balance'
        else:
            confidence = 'C' # 見送り
            strategy_type = 'skip'
            
        race_strategies.append({
            'race_id': race_id,
            'race_name': race_name,
            'confidence': confidence,
            'strategy': strategy_type,
            'top1': top1,
            'top2': top2,
            'top3': top3,
            'others': others
        })

    # Best 3 (S/A/Bのみ対象)
    best_races = [r for r in race_strategies if r['confidence'] in ['S', 'A', 'B']]
    best_races = sorted(best_races, key=lambda x: x['top1']['score'], reverse=True)[:3]
    
    if best_races:
        output_lines.append("\n## 🔥 本日の勝負レース Best 3\n")
        output_lines.append("自信度が高いレースです。資金を厚めに配分することを推奨します。\n")
        for i, race in enumerate(best_races):
            h1, h2 = race['top1'], race['top2']
            output_lines.append(f"### {i+1}. {race['race_name']} (自信度: {race['confidence']})")
            output_lines.append(f"- **◎ 本命**: {int(h1['馬番'])} {h1['馬名']} (Score: {h1['score']:.3f})")
            output_lines.append(f"- **○ 対抗**: {int(h2['馬番'])} {h2['馬名']}")
            output_lines.append(f"- **推奨**: 単勝 {int(h1['馬番'])}, 馬連 {int(h1['馬番'])}-{int(h2['馬番'])}\n")

    # All Races
    output_lines.append("## 📋 全レース買い目リスト\n")
    for race in race_strategies:
        idx = str(race['race_id'])
        name = race['race_name']
        conf = race['confidence']
        h1 = race['top1']
        h2 = race['top2']
        h3 = race['top3']
        
        output_lines.append(f"### ID:{idx} {name} [{conf}]")
        output_lines.append(f"- ◎ {int(h1['馬番'])} {h1['馬名']}")
        output_lines.append(f"- ○ {int(h2['馬番'])} {h2['馬名']}")
        output_lines.append(f"- ▲ {int(h3['馬番'])} {h3['馬名']}")
        
        bets = f"単勝 {int(h1['馬番'])}"
        if race['strategy'] == 'winner': bets += " (一点勝負)"
        output_lines.append(f"- 買い目: {bets}, 馬連 {int(h1['馬番'])}-{int(h2['馬番'])}\n")

    output_md = f"buy_list_{target_date.replace('/', '')}.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    csv_out = df[['race_num', '馬番', '馬名', 'race_type', 'course_len', 'score']].copy()
    csv_out.to_csv('prediction_20260125.csv', index=True, encoding='utf-8-sig')
    
    print(f"完了。{output_md} と prediction_20260125.csv を作成しました。")

if __name__ == '__main__':
    predict_tomorrow()
