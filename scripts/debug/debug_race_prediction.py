
import os
import sys
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import pickle

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import HEADERS, MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.scraping import Shutuba, Odds
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor

def load_prediction_pipeline():
    """モデルと前処理パイプラインをロード"""
    print("Loading models...")
    
    # Check for ensemble model first
    if os.path.exists(os.path.join(MODEL_DIR, 'model_lgbm_0.pkl')):
        model = EnsembleModel()
        model.load(MODEL_DIR)
        print("Ensemble model loaded.")
    else:
        model = HorseRaceModel()
        model.load()
        print(f"Single model loaded: {type(model)}")

    # Processor / Engineer
    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    print("Processor loaded.")
        
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
    print("Engineer loaded.")
        
    return RacePredictor(model, processor, engineer)

def load_historical_data():
    """過去データをロード（特徴量生成用）"""
    print("Loading historical data...")
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    horse_results = None
    peds = None
    
    if os.path.exists(hr_path):
        with open(hr_path, 'rb') as f:
            horse_results = pickle.load(f)
        print(f"Horse Results loaded: {len(horse_results)} records")
            
    if os.path.exists(peds_path):
        with open(peds_path, 'rb') as f:
            peds = pickle.load(f)
        print(f"Peds loaded: {len(peds)} records")
            
    return horse_results, peds

def debug_race(race_id):
    print(f"\n--- Debugging Race {race_id} ---", flush=True)
    
    try:
        predictor = load_prediction_pipeline()
        horse_results, peds = load_historical_data()
        
        # 1. Scrape Shutuba
        print("Scraping shutuba...", flush=True)
        df_shutuba = Shutuba.scrape(race_id)
        if df_shutuba.empty:
            print("ERROR: Shutuba is empty.", flush=True)
            return
            
        print("Shutuba Data (Head):", flush=True)
        print(df_shutuba.columns.tolist(), flush=True)
        try:
            print(df_shutuba[['馬番', '馬名', '性齢', '斤量', '騎手', '厩舎', '父', '母']].head(), flush=True)
        except KeyError as e:
            print(f"KeyError displaying head: {e}", flush=True)
        
        # Cleaning list values
        for col in df_shutuba.columns:
            if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
                print(f"Flattening list column: {col}", flush=True)
                df_shutuba[col] = df_shutuba[col].apply(lambda x: str(x[0]) if isinstance(x, list) and len(x)>0 else x)

        df_shutuba['date'] = pd.to_datetime('2026-02-16') # 仮の日付

        # 3.1 Processor
        print("\nRunning processor.process_results...", flush=True)
        try:
            df_processed = predictor.processor.process_results(df_shutuba)
            print("Processed Columns:", df_processed.columns.tolist(), flush=True)
            print("Sample processed data (weight, weight_diff):", flush=True)
            if '体重' in df_processed.columns:
                print(df_processed[['体重', '体重変化']].head(), flush=True)
            else:
                print("'体重' column missing!", flush=True)
        except Exception as e:
             print(f"Error in process_results: {e}", flush=True)
             import traceback
             traceback.print_exc()
             return

        # 3.2 Impute Weight
        if '体重' in df_processed.columns:
            mean_weight = df_processed['体重'].mean()
            if pd.isna(mean_weight): mean_weight = 470.0
            df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
        if '体重変化' in df_processed.columns:
            df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

        # 3.3 Engineer (History)
        print("\nAdding history features...", flush=True)
        try:
            if horse_results is not None:
                # Check index type
                print(f"History Index Type: {horse_results.index.dtype}", flush=True)
                # Check if horse names match
                match_count = df_shutuba['馬名'].isin(horse_results.index).sum()
                print(f"Matches in history: {match_count} / {len(df_shutuba)}", flush=True)
                
                df_processed = predictor.engineer.add_horse_history_features(df_processed, horse_results)
                print("History added. Columns:", [c for c in df_processed.columns if 'avg_' in c or 'max_' in c][:5], flush=True)
            else:
                print("Warning: No historical data loaded.", flush=True)
        except Exception as e:
             print(f"Error in add_horse_history_features: {e}", flush=True)
             import traceback
             traceback.print_exc()
             # Continue if possible? Maybe not.
             return
        
        try:
            # Generate Jockey Stats from history if available
            jockey_stats = None
            if horse_results is not None:
                print("Generating jockey stats from history...", flush=True)
                print(f"Horse Results Columns: {horse_results.columns.tolist()}", flush=True)
                if 'jockey_id' not in horse_results.columns:
                     print("CRITICAL: jockey_id missing in horse_results!", flush=True)
                
                # 全履歴を使って統計生成
                _, jockey_stats = predictor.engineer.add_jockey_features(horse_results, jockey_stats=None)
                
                if jockey_stats is not None:
                    print(f"Jockey stats generated: {len(jockey_stats)} jockeys", flush=True)
                else:
                    print("Jockey stats generation failed (None returned).", flush=True)

            df_processed = predictor.engineer.add_course_suitability_features(df_processed, horse_results)
            
            # Apply jockey stats
            df_processed, _ = predictor.engineer.add_jockey_features(df_processed, jockey_stats=jockey_stats)
            
            if peds is not None:
                df_processed = predictor.engineer.add_pedigree_features(df_processed, peds)
            
            df_processed = predictor.engineer.add_odds_features(df_processed)
        except Exception as e:
             print(f"Error in other engineering steps: {e}", flush=True)
             import traceback
             traceback.print_exc()

        # 3.4 Encode
        cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
        cat_cols = [c for c in cat_cols if c in df_processed.columns]
        df_processed = predictor.processor.encode_categorical(df_processed, cat_cols)
        
        # 3.5 Feature Selection
        feature_names = predictor.model.feature_names
        print(f"\nModel expects {len(feature_names)} features.", flush=True)
        
        X = pd.DataFrame(index=df_processed.index)
        missing_cols = []
        for col in feature_names:
            if col in df_processed.columns:
                X[col] = df_processed[col]
            else:
                X[col] = 0
                missing_cols.append(col)
                
        if missing_cols:
            print(f"WARNING: Missing columns ({len(missing_cols)}): {missing_cols[:5]}...", flush=True)
        
        # Check for all zeros or constant
        print("\nChecking X stats:", flush=True)
        desc = X.describe().T
        print(desc[['mean', 'std', 'min', 'max']].head(10), flush=True)
        
        # Check if any row is all zeros? No, numeric_X median fill happens next
        numeric_X = X.select_dtypes(include=[np.number])
        X[numeric_X.columns] = numeric_X.fillna(numeric_X.median())
        X = X.fillna(0)
        
        if predictor.processor.scaler:
            X = predictor.processor.transform_scale(X)

        # Check Encoded Categorical Values
        print("\n--- Encoded Categorical Values ---", flush=True)
        cat_cols_to_check = ['厩舎', 'trainer_id', '騎手', 'jockey_id', '父', 'sire', '母', 'dam', '馬番']
        for col in cat_cols_to_check:
            if col in X.columns:
                print(f"{col}: {X[col].head().tolist()}", flush=True)
            else:
                 pass # print(f"{col}: Not in X")

        # Feature Importance
        print("\n--- Feature Importance (Top 20) ---", flush=True)
        try:
            # Check if EnsembleModel
            if isinstance(predictor.model, EnsembleModel):
                fi_df = predictor.model.get_feature_importance(top_n=20)
                print(fi_df, flush=True)
            elif hasattr(predictor.model.model, 'feature_importance'):
                # LightGBM (Single)
                imp = predictor.model.model.feature_importance(importance_type='gain')
                names = predictor.model.feature_names
                fi_df = pd.DataFrame({'feature': names, 'importance': imp})
                fi_df = fi_df.sort_values('importance', ascending=False).head(20)
                print(fi_df, flush=True)
            elif hasattr(predictor.model.model, 'feature_importances_'):
                 # Sklearn (Single)
                imp = predictor.model.model.feature_importances_
                names = predictor.model.feature_names
                fi_df = pd.DataFrame({'feature': names, 'importance': imp})
                fi_df = fi_df.sort_values('importance', ascending=False).head(20)
                print(fi_df, flush=True)
        except Exception as e:
            print(f"Could not get feature importance: {e}", flush=True)

        # 4. Predict
        print("\nPredicting...", flush=True)
        probs = predictor.model.predict(X)
        
        print("\n--- Predictions ---", flush=True)
        results = pd.DataFrame({
            'Horse': df_shutuba['馬名'],
            'Number': df_shutuba['馬番'],
            'Score': probs
        })
        # Add rank
        results['Rank'] = results['Score'].rank(ascending=False).astype(int)
        results = results.sort_values('Score', ascending=False)
        print(results, flush=True)
        
        # Check correlation with Number
        results['NumInt'] = results['Number'].astype(int)
        corr = results['Score'].corr(results['NumInt'])
        print(f"\nCorrelation with Horse Number: {corr:.4f}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR in debug_race: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_race('202605010611')
