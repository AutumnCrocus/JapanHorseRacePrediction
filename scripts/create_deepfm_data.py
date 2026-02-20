
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import PROCESSED_DATA_DIR

DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/models/deepfm')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'deepfm_data.pkl')

def create_deepfm_data():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'rb') as f:
        data_dict = pickle.load(f)
        df = data_dict['data']

    # Ensure race_id is a column
    if 'race_id' not in df.columns:
        if 'original_race_id' in df.columns:
            df['race_id'] = df['original_race_id']
        elif df.index.name == 'race_id':
             df = df.reset_index()
        else:
             df = df.reset_index().rename(columns={'index': 'race_id'})

    print(f"Loaded DataFrame shape: {df.shape}")
    print("Columns sample:", df.columns[:20].tolist())

    # --- Feature Mapping (Japanese -> English) ---
    # Rename columns to standard names for DeepFM
    rename_map = {
        '斤量': 'burden_weight',
        '体重': 'weight',
        '体重変化': 'weight_diff',
        '馬番': 'horse_number',
        '枠番': 'frame_number',
        '年齢': 'age',
        '性': 'sex',
        'weather': 'weather', # Already English
        'ground_state': 'ground_state', # Already English
        'race_type': 'race_type', # Already English
        'venue_id': 'venue_id',
        'course_len': 'course_len'
    }
    
    # Handle optional ID columns
    if 'jockey_id' not in df.columns and '騎手' in df.columns:
        df['jockey_id'] = df['騎手']
    if 'trainer_id' not in df.columns and '調教師' in df.columns:
        df['trainer_id'] = df['調教師']
        
    for jp, en in rename_map.items():
        if jp in df.columns:
            df[en] = df[jp]
        else:
            print(f"Warning: Column {jp} not found.")
            # Set default if critical
            if en in ['age']: df[en] = 4
            if en in ['burden_weight']: df[en] = 55
            if en in ['weight']: df[en] = 480
            if en in ['weight_diff']: df[en] = 0
            if en in ['horse_number']: df[en] = 8

    # --- Feature Selection ---
    
    # Sparse Features (Categorical)
    sparse_features = [
        'horse_id', 'jockey_id', 'trainer_id',
        'sex', 'weather', 'ground_state', 'race_type', 'venue_id', 'frame_number'
    ]
    # Pedigree (sire, dam)
    possible_ped_cols = ['sire', 'dam', 'sire_id', 'dam_id']
    for c in possible_ped_cols:
        if c in df.columns:
            sparse_features.append(c)
            
    sparse_features = [c for c in sparse_features if c in df.columns]
    # Ensure critical cold start features are present
    for req in ['sire', 'dam', 'trainer_id']:
        if req not in sparse_features and req in df.columns:
             sparse_features.append(req)
             
    sparse_features = list(set(sparse_features)) # unique
    print(f"Sparse Features: {sparse_features}")

    # Dense Features (Numerical)
    dense_features = [
        'burden_weight', 'weight', 'weight_diff', 'age', 'horse_number', 'course_len'
    ]
    if 'n_horses' in df.columns: 
        dense_features.append('n_horses')
    if '頭数' in df.columns:
        df['n_horses'] = df['頭数']
        if 'n_horses' not in dense_features: dense_features.append('n_horses')
        
    dense_features = [c for c in dense_features if c in df.columns]
    print(f"Dense Features: {dense_features}")

    if not sparse_features and not dense_features:
        print("Error: No features selected!")
        return

    # Target
    target = ['target']

    # Preserve original horse_number for metadata
    if 'horse_number' in df.columns:
        df['horse_number_original'] = df['horse_number']
    else:
        df['horse_number_original'] = 0

    # --- Preprocessing ---
    # 1. Fill NA
    for feat in sparse_features:
        df[feat] = df[feat].fillna('unknown').astype(str)
    
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)

    # 2. Label Encoding (Sparse)
    label_encoders = {}
    sparse_features_config = []
    
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
        label_encoders[feat] = lbe
        sparse_features_config.append({
            'name': feat,
            'vocabulary_size': int(df[feat].max() + 1),
            'embedding_dim': 8
        })
    
    # 3. MinMax Scaling (Dense)
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_features] = mms.fit_transform(df[dense_features])
    
    dense_features_config = [{'name': feat, 'dim': 1} for feat in dense_features]

    # --- Split Data ---
    print("Splitting train/test...")
    if 'year' not in df.columns:
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        else:
            df['year'] = df['race_id'].astype(str).str[:4].astype(int)

    train_df = df[df['year'] <= 2024]
    test_df = df[df['year'] == 2025]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    feature_names = sparse_features + dense_features
    
    train_model_input = {name: train_df[name].values for name in feature_names}
    test_model_input = {name: test_df[name].values for name in feature_names}
    
    train_target = train_df[target].values
    test_target = test_df[target].values 

    # --- Save ---
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Select columns for metadata (training data needs race_id/horse_number for stacking)
    meta_cols = ['race_id', 'horse_id', 'horse_number_original', 'target', 'year']
    if 'date' in df.columns: meta_cols.append('date')
    
    save_data = {
        'train_model_input': train_model_input,
        'train_target': train_target,
        'test_model_input': test_model_input,
        'test_target': test_target,
        'feature_config': {
            'sparse': sparse_features_config,
            'dense': dense_features_config
        },
        'label_encoders': label_encoders,
        # Save metadata for both train and test to allow full scoring
        'test_df': test_df[meta_cols].copy(),
        'train_df': train_df[meta_cols].copy() 
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved DeepFM data to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_deepfm_data()
