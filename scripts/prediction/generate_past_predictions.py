
import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# プロジェクトルート
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE

# 設定
TARGET_YEAR = 2023
TARGET_MONTH = 12 # 2023年12月のデータを対象
RANKING_MODEL_DIR = os.path.join(MODEL_DIR, "standalone_ranking")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'prediction_202412_ltr.csv')

def load_resources():
    print("Loading data...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    
    # Pre-processing
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    # Filter by date (YYYYMM) -> Random Sample for testing
    # race_id: YYYY... 
    # prefix = f"{TARGET_YEAR}{TARGET_MONTH:02d}"
    # df_target = results[results['race_id'].astype(str).str.startswith(prefix)].copy()
    
    # 存在するレースIDからランダムに100件抽出 (2020年以降に限定してみる)
    all_rids = results['race_id'].astype(str)
    # 2020年以降
    recent_rids = all_rids[all_rids.str.startswith(('2020', '2021', '2022', '2023', '2024', '2025'))]
    
    if len(recent_rids) > 20:
        target_rids = recent_rids.sample(20, random_state=42).tolist()
    else:
        target_rids = recent_rids.tolist() if len(recent_rids) > 0 else all_rids.sample(20, random_state=42).tolist()
        
    df_target = results[results['race_id'].astype(str).isin(target_rids)].copy()
    
    print(f"Sampled {len(df_target)} rows from {len(target_rids)} races.")
    return df_target, hr, peds

def load_model(model_dir):
    print(f"Loading model from {model_dir}...")
    model_path = os.path.join(model_dir, "ranking_model.pkl")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        model = data.get('model', data)
        features = data.get('feature_names', [])
    else:
        model = data
        features = []
        
    proc_path = os.path.join(model_dir, 'processor.pkl')
    eng_path = os.path.join(model_dir, 'engineer.pkl')
    
    if not os.path.exists(proc_path): proc_path = os.path.join(MODEL_DIR, 'processor.pkl')
    if not os.path.exists(eng_path): eng_path = os.path.join(MODEL_DIR, 'engineer.pkl')

    with open(proc_path, 'rb') as f: processor = pickle.load(f)
    with open(eng_path, 'rb') as f: engineer = pickle.load(f)
    
    return model, features, processor, engineer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def main():
    df_raw, hr, peds = load_resources()
    model, features, processor, engineer = load_model(RANKING_MODEL_DIR)
    
    race_ids = df_raw['race_id'].unique()
    print(f"Target Races: {len(race_ids)}")
    
    all_preds = []
    
    for rid in tqdm(race_ids):
        race_df = df_raw[df_raw['race_id'] == rid].copy()
        
        # Feature Engineering
        try:
            df_proc = processor.process_results(race_df)
            df_proc = engineer.add_horse_history_features(df_proc, hr)
            df_proc = engineer.add_course_suitability_features(df_proc, hr)
            df_proc, _ = engineer.add_jockey_features(df_proc)
            df_proc = engineer.add_pedigree_features(df_proc, peds)
            df_proc = engineer.add_odds_features(df_proc)
            
            # Prediction
            X = pd.DataFrame(index=df_proc.index)
            feats = features if features else [c for c in df_proc.select_dtypes(include=[np.number]).columns 
                                             if c not in ['rank', 'probability', '馬番']]
            for c in feats:
                if c in df_proc.columns:
                    X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
                else:
                    X[c] = 0
            
            # LTR Score
            scores = model.predict(X)
            
            # Create Result DataFrame
            df_res = df_proc[['race_id', '馬番', '馬名']].copy()
            df_res['ltr_score'] = scores
            df_res['horse_number'] = df_res['馬番']
            df_res['horse_name'] = df_res['馬名']
            
            # Probability (Softmax within race)
            df_res['probability'] = softmax(scores)
            
            # Odds (単勝オッズがあれば取得、なければ1.0)
            if '単勝' in df_proc.columns:
                df_res['win_odds'] = pd.to_numeric(df_proc['単勝'], errors='coerce').fillna(0.0)
            else:
                df_res['win_odds'] = 0.0
                
            # Rank (Actual Rank if available)
            if '着順' in df_proc.columns:
                 df_res['rank_prediction'] = pd.to_numeric(df_proc['着順'], errors='coerce').fillna(99) # 名前は互換性維持のため rank_prediction とするが中身は実際の結果かもしれない
            
            all_preds.append(df_res)
            
        except Exception as e:
            print(f"Error processing race {rid}: {e}")
            continue

    if all_preds:
        final_df = pd.concat(all_preds)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved predictions to {OUTPUT_CSV}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()
