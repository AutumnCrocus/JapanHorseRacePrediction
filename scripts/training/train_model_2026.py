
"""
2026年実験モデル学習スクリプト
- 対象期間: 2010-2023 (Train), 2024 (Val)
- アルゴリズム: LightGBM
- 入力: data/processed/dataset_2010_2025.pkl
- 出力: models/experiment_model_2026.pkl
"""
import os
import sys
import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import MODEL_DIR
from modules.training import HorseRaceModel

# データセットパス
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset_2010_2025.pkl')
# 保存先
MODEL_NAME = "experiment_model_2026.pkl"
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def train_model():
    print("=== Training Experiment Model 2026 ===")
    
    # 1. データセット読み込み
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found. Run create_dataset_2025.py first.")
        return
        
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
        
    df_proc = dataset['data']
    feature_names = dataset['feature_names']
    
    print("DEBUG: df_proc columns:", df_proc.columns[:10])
    print("DEBUG: df_proc index name:", df_proc.index.name)
    print("DEBUG: df_proc index names:", df_proc.index.names)
    
    # --- DeepFM Score Integration ---
    deepfm_path = os.path.join(os.path.dirname(__file__), '../../data/processed/deepfm_scores.csv')
    if os.path.exists(deepfm_path):
        print(f"Loading DeepFM scores from {deepfm_path}...")
        deepfm_df = pd.read_csv(deepfm_path)
        
        # Prepare horse_number and race_id mapping
        if 'horse_number' not in df_proc.columns and '馬番' in df_proc.columns:
             df_proc['horse_number'] = df_proc['馬番']
        
        if 'horse_number' not in df_proc.columns:
             print("Warning: 'horse_number' column missing in dataset. Cannot merge DeepFM scores.")
        else:
            # Identify race identifier column (prefer original_race_id if race_id is missing)
            race_id_col = 'race_id'
            if 'race_id' not in df_proc.columns:
                 if 'original_race_id' in df_proc.columns:
                     race_id_col = 'original_race_id'
                 elif df_proc.index.name == 'race_id':
                     df_proc = df_proc.reset_index()
                     race_id_col = 'race_id'
                 else:
                     # Check if unnamed index looks like race_id (12 digit numeric)
                     sample_idx = str(df_proc.index[0])
                     if len(sample_idx) >= 10 and sample_idx.isdigit():
                          df_proc = df_proc.reset_index().rename(columns={'index': 'race_id'})
                          race_id_col = 'race_id'
            
            if race_id_col in df_proc.columns:
                 df_proc[race_id_col] = df_proc[race_id_col].astype(str)
                 df_proc['horse_number'] = df_proc['horse_number'].astype(int)
                 deepfm_df['race_id'] = deepfm_df['race_id'].astype(str)
                 deepfm_df['horse_number'] = deepfm_df['horse_number'].astype(int)
                 
                 # Merge (Left Join to keep all training data)
                 df_proc = pd.merge(df_proc, deepfm_df[['race_id', 'horse_number', 'deepfm_score']], 
                                    left_on=[race_id_col, 'horse_number'], 
                                    right_on=['race_id', 'horse_number'], how='left')
                 
                 # Fill NA with mean (neutral) for horses not in DeepFM data
                 if 'deepfm_score' in df_proc.columns:
                     mean_score = df_proc['deepfm_score'].mean()
                     if pd.isna(mean_score): mean_score = 0
                     df_proc['deepfm_score'] = df_proc['deepfm_score'].fillna(mean_score)
                     
                     print(f"Merged DeepFM scores. Added feature 'deepfm_score'.")
                     if 'deepfm_score' not in feature_names:
                         feature_names.append('deepfm_score')
            else:
                 print("Warning: Could not identify race_id in index or columns. Skipping DeepFM merge.")
    else:
        print("Warning: DeepFM scores file not found. Training without it.")
    # -------------------------------
    
    print(f"Total samples: {len(df_proc)}")
    print(f"Features: {len(feature_names)}")

    # 2. データ分割
    # Train: 2010-2023
    # Val: 2024
    # Test: 2025 (Not used for training, kept for backtest)
    
    print("Splitting data...")
    
    # 特徴量フィルタリング (数値型のみ & 除外カラム指定)
    exclude_cols = [
        'race_id', 'horse_name', 'jockey_name', 'trainer_name', 'horse_id', 'date', 'original_race_id', 'year',
        '着順', '着 順', 'rank_num', 'target',
        'タイム', 'タイム秒', '上り', '通過', 'running_style', # These are results
        '単勝', '単 勝', '人気', '人 気', # These are results
        '戦績', '賞金', '賞金（万円）', 'タイム指数', '着差', # Leaks
        'is_win', 'is_place', 'return', # Critical Leaks
        '性齢', '騎手', '調教師', '馬主', 'ブリンカー', '斤量', # Raw string cols
        '馬体重', '調教タイム', '厩舎コメント', '備考', 'sire', 'dam', 'jockey_id', 'trainer_id'
    ]
    
    # データセットのfeature_namesから、実際に使用可能な数値カラムのみを抽出
    real_feature_names = []
    for col in feature_names:
        if col in exclude_cols:
            continue
        if col not in df_proc.columns:
            continue
        # Check dtype
        if pd.api.types.is_numeric_dtype(df_proc[col]):
            real_feature_names.append(col)
        else:
            # Try converting to numeric, if fail, skip
            try:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='raise')
                real_feature_names.append(col)
            except:
                # print(f"Skipping non-numeric column: {col}")
                pass
                
    feature_names = real_feature_names
    print(f"Selected features: {len(feature_names)}")

    train_mask = (df_proc['year'] <= 2023)
    val_mask = (df_proc['year'] == 2024)

    
    X_train = df_proc.loc[train_mask, feature_names].fillna(0)
    y_train = df_proc.loc[train_mask, 'target']
    
    X_val = df_proc.loc[val_mask, feature_names].fillna(0)
    y_val = df_proc.loc[val_mask, 'target']
    
    print(f"Train: {len(X_train)} (2010-2023)")
    print(f"Val:   {len(X_val)} (2024)")
    
    # 3. モデル学習
    print("Training LightGBM...")
    model = HorseRaceModel(model_type='lgbm')
    
    # パラメータ調整（必要に応じて）
    # model.model_params['n_estimators'] = 1000
    # model.model_params['learning_rate'] = 0.03
    
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    print(f"Metrics: {metrics}")
    
    # 4. 保存
    model.save(SAVE_PATH)
    
    print(f"Model saved to {SAVE_PATH}")
    print("Done.")

if __name__ == "__main__":
    train_model()
