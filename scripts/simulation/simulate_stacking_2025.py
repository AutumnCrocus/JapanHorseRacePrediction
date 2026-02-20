
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.training import HorseRaceModel
from modules.constants import PROCESSED_DATA_DIR, MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2026.pkl')
DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
DEEPFM_SCORES_PATH = os.path.join(PROCESSED_DATA_DIR, 'deepfm_scores.csv')

def simulate():
    print("=== Stacking Model Simulation (2025) ===")
    
    print(f"Loading model form {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_model_2026.py first.")
        return

    try:
        model = HorseRaceModel()
        model.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Try loading raw pickle if HorseRaceModel load fails (compatibility)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

    print("Loading dataset...")
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
    df = dataset['data']
    
    # 2025 only
    if 'year' not in df.columns:
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        else:
            # Try to infer from race_id or other means, or just use last N records
            pass
            
    df = df[df['year'] == 2025].copy()
    print(f"2025 records: {len(df)}")
    
    print("Loading DeepFM scores...")
    if os.path.exists(DEEPFM_SCORES_PATH):
        scores = pd.read_csv(DEEPFM_SCORES_PATH)
        scores['race_id'] = scores['race_id'].astype(str)
        scores['horse_number'] = scores['horse_number'].astype(int)
        
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
        scores['race_id'] = scores['race_id'].astype(str)
        
        # Merge
        df = pd.merge(df, scores[['race_id', 'horse_number', 'deepfm_score']], 
                      left_on=[race_id_col, 'horse_number'], 
                      right_on=['race_id', 'horse_number'], how='left')
        
        # Fill NA
        if 'deepfm_score' in df.columns:
            mean_val = df['deepfm_score'].mean()
            if pd.isna(mean_val): mean_val = 0
            df['deepfm_score'] = df['deepfm_score'].fillna(mean_val)
            print("DeepFM scores merged.")
    else:
        print("Warning: DeepFM scores not found!")

    # Predict
    print("Predicting...")
    
    # Ensure all features exist
    # Access feature_names from model attribute or if it's a raw lgb model
    feature_names = getattr(model, 'feature_names', None)
    if feature_names is None:
        # If it's pure LGBMBooster
        feature_names = model.feature_name()
    
    print(f"Model uses {len(feature_names)} features.")
    
    # Missing columns check
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing[:5]}...")
        for c in missing:
            df[c] = 0
            
    X = df[feature_names].fillna(0)
    
    # Predict
    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
    else:
        # LGBM Booster
        y_pred = model.predict(X)
        
    df['pred'] = y_pred
    
    # Calculate ROI (Single Win)
    print("Calculating ROI...")
    races = df.groupby('race_id')
    invest = 0
    return_amount = 0
    hits = 0
    
    # Odds checks
    odds_col = None
    if '単勝' in df.columns: odds_col = '単勝'
    elif 'win_odds' in df.columns: odds_col = 'win_odds'
    
    rank_col = None
    if '着順' in df.columns: rank_col = '着順'
    elif 'rank' in df.columns: rank_col = 'rank'
    elif 'target' in df.columns: rank_col = 'target' # target might be 1/0
    
    for rid, group in races:
        invest += 100
        # Sort by pred desc
        top1 = group.sort_values('pred', ascending=False).iloc[0]
        
        is_hit = False
        if rank_col == 'target':
            if top1[rank_col] == 1: is_hit = True
        elif rank_col:
            try:
                r = int(top1[rank_col])
                if r == 1: is_hit = True
            except: pass
            
        if is_hit:
            hits += 1
            if odds_col:
                try:
                    o = float(top1[odds_col])
                    return_amount += o * 100
                except:
                    return_amount += 100 # Fallback
            else:
                 return_amount += 100 # No odds info
                
    roi = return_amount / invest * 100 if invest > 0 else 0
    print(f"Races: {len(races)}")
    print(f"Hits: {hits} ({hits/len(races)*100:.1f}%)")
    print(f"Return: {int(return_amount)} / {invest}")
    print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    simulate()
