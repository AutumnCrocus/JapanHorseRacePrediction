
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import DataProcessor, FeatureEngineer
from scripts.train_ranking_standalone import prepare_ranking_labels

def main():
    MODEL_PATH = os.path.join(MODEL_DIR, "standalone_ranking", "ranking_model.pkl")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    print("評価用のデータをロード中...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)

    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
    
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()

    # 2025年のデータのみ抽出
    df_2025 = results[results['date'].dt.year == 2025].copy()
    if len(df_2025) == 0:
        print("No data for 2025.")
        return

    print(f"2025年のレース数: {len(df_2025['race_id'].unique())}")

    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 特徴量生成 (1レースずつ行うと遅いので一括で行う)
    print("特徴量生成中 (2025年分)...")
    df_proc = processor.process_results(df_2025)
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # ターゲット: 関連度スコア
    df_proc['relevance'] = prepare_ranking_labels(df_proc)
    
    # リーク防止
    exclude_cols = [
        'rank', 'date', 'race_id', 'horse_id', 'target', '着順', 'relevance',
        'time', '着差', '通過', '上り', '単勝', '人気', 'horse_name', 'jockey', 
        'trainer', 'owner', 'gender', 'original_race_id', 'タイム', 'タイム秒',
        '着 順', '不正', '失格', '中止', '取消', '除外', 'running_style',
        '体重', '体重変化', '馬体重', '単 勝', '人 気', '賞金', '賞金（万円）',
        '付加賞（万円）', 'rank_num', 'is_win', 'is_place', 'last_3f_num',
        'odds', 'popularity', 'return'
    ]
    
    # 予測
    print("予測中...")
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    X = df_proc[feature_names]
    df_proc['ranking_score'] = model.predict(X)
    
    # レースごとに評価
    print("スコア計算中...")
    ndcgs = []
    top1_hits = 0
    top3_hits = 0
    total_races = 0

    for race_id, group in df_proc.groupby('race_id'):
        if len(group) < 2: continue
        
        y_true = group['relevance'].values.reshape(1, -1)
        y_score = group['ranking_score'].values.reshape(1, -1)
        
        # NDCG@5
        if y_true.sum() > 0:
            ndcg = ndcg_score(y_true, y_score, k=5)
            ndcgs.append(ndcg)
        
        # 1位的中率
        top_idx = group['ranking_score'].argmax()
        actual_rank = group.iloc[top_idx]['rank'] if 'rank' in group.columns else group.iloc[top_idx]['着順']
        if actual_rank == 1:
            top1_hits += 1
        if actual_rank <= 3:
            top3_hits += 1
            
        total_races += 1

    print(f"\n=== 2025年検証結果 ({total_races} レース) ===")
    print(f"平均 NDCG@5: {np.mean(ndcgs):.4f}")
    print(f"1位的中率: {top1_hits/total_races:.2%} ({top1_hits}/{total_races})")
    print(f"3着内率 (予測1位): {top3_hits/total_races:.2%} ({top3_hits}/{total_races})")

if __name__ == "__main__":
    main()
