
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import PROCESSED_DATA_DIR

DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
DEEPFM_SCORES_PATH = os.path.join(PROCESSED_DATA_DIR, 'deepfm_scores.csv')

def verify_cold_start():
    print("=== Cold Start Verification ===")
    
    # 1. Load Data
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    df = dataset['data']
    
    print(f"Loading scores from {DEEPFM_SCORES_PATH}...")
    if not os.path.exists(DEEPFM_SCORES_PATH):
        print("Scores file not found.")
        return
        
    scores_df = pd.read_csv(DEEPFM_SCORES_PATH)
    scores_df['race_id'] = scores_df['race_id'].astype(str)
    scores_df['horse_number'] = scores_df['horse_number'].astype(int)
    
    # Merge scores
    # Ensure keys match
    if 'horse_number' not in df.columns and '馬番' in df.columns:
         df['horse_number'] = df['馬番']
         
    if 'horse_number' not in df.columns:
         print("Error: horse_number column missing.")
         return
         
    # Identify race identifier column
    race_id_col = 'race_id'
    if 'race_id' not in df.columns:
         if 'original_race_id' in df.columns:
             race_id_col = 'original_race_id'
         else:
             df = df.reset_index()
             if 'race_id' not in df.columns and 'index' in df.columns:
                 df = df.rename(columns={'index': 'race_id'})
             
             if 'race_id' in df.columns:
                 race_id_col = 'race_id'
             else:
                 print("Error: Could not identify race_id column in dataset.")
                 return

    df[race_id_col] = df[race_id_col].astype(str)
    scores_df['race_id'] = scores_df['race_id'].astype(str)
    
    # Merge
    df = pd.merge(df, scores_df[['race_id', 'horse_number', 'deepfm_score']], 
                  left_on=[race_id_col, 'horse_number'], 
                  right_on=['race_id', 'horse_number'], how='inner')
    
    print(f"Merged Data: {len(df)} rows")
    
    # 2. Identify New Horses in 2025
    # Definition: Horses that appear in 2025 but NOT in 2010-2024
    
    # Ensure year column
    if 'year' not in df.columns:
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        else:
            df['year'] = df['race_id'].astype(str).str[:4].astype(int)
            
    train_horses = set(df[df['year'] <= 2024]['horse_id'].unique())
    test_df = df[df['year'] == 2025]
    test_horses = set(test_df['horse_id'].unique())
    
    new_horses = test_horses - train_horses
    print(f"Total horses in 2025: {len(test_horses)}")
    print(f"New horses (Cold Start) in 2025: {len(new_horses)}")
    
    if len(new_horses) == 0:
        print("No new horses found? Check data range.")
        return

    # 3. Analyze Scores for New Horses
    new_horse_rows = test_df[test_df['horse_id'].isin(new_horses)]
    cold_scores = new_horse_rows['deepfm_score']
    
    print("\n--- Cold Start Scores Statistics ---")
    print(cold_scores.describe())
    
    # Check variance
    std = cold_scores.std()
    print(f"Standard Deviation: {std:.4f}")
    
    if std < 0.0001:
        print("FAIL: Scores have almost no variance. Cold start handling might be ineffective (default values used).")
    else:
        print("PASS: Scores show variance. Features like Pedigree/Trainer are likely working.")
        
    # Validating Feature Importance (Pseudo)
    # If all scores are same, it means id embeddings (which are unknown) dominated and other features were ignored or same.
    
    # Save histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cold_scores, bins=50, alpha=0.7, label='New Horses')
    plt.title('DeepFM Score Distribution for Cold Start Horses (2025)')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('cold_start_distribution.png')
    print("Saved histogram to cold_start_distribution.png")

if __name__ == "__main__":
    verify_cold_start()
