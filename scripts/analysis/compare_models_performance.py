
import os
import sys
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add current directory to path
sys.path.append(os.getcwd())

from modules.constants import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
    RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
)
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def load_data():
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        horse_results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    return results, horse_results, peds

def main():
    print("Loading data...")
    results, horse_results, peds = load_data()
    
    # Split into train (before 2025) and eval (2025)
    results_train = results[~results.index.astype(str).str.startswith('2025')]
    results_eval = results[results.index.astype(str).str.startswith('2025')]
    
    print(f"Training data: {len(results_train)} rows")
    print(f"Evaluation data (2025): {len(results_eval)} rows")
    
    # Target algorithms
    algos = ['lgbm', 'rf', 'pytorch_mlp']
    results_list = []
    
    # Preprocessing (common for tree models, might need different for MLP if scale=True)
    # Actually, we should call prepare_training_data twice if we want different scaling
    
    for algo in algos:
        print(f"\n--- Training {algo} ---")
        scale = (algo == 'pytorch_mlp')
        
        # We need to prepare data for training (excluding 2025)
        X_train, y_train, processor, engineer = prepare_training_data(
            results_train, horse_results, peds, scale=scale
        )
        
        model = HorseRaceModel(model_type=algo)
        start_time = time.time()
        metrics_train = model.train(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f}s")
        
        # Prepare evaluation data (using same processor/engineer)
        # Note: We need a way to pass existing processor to prepare_training_data 
        # but prepare_training_data creates a new one.
        # Let's manually process evaluation data.
        
        print(f"Evaluating {algo} on 2025 data...")
        df_eval = processor.process_results(results_eval)
        
        # Copy-paste logic from evaluate_model in main.py to be consistent
        horse_results_tmp = horse_results.copy()
        horse_results_tmp.columns = horse_results_tmp.columns.str.replace(' ', '')
        if '着順' in horse_results_tmp.columns:
            horse_results_tmp['着順'] = pd.to_numeric(horse_results_tmp['着順'], errors='coerce')
            
        df_eval = engineer.add_horse_history_features(df_eval, horse_results_tmp)
        df_eval = engineer.add_course_suitability_features(df_eval, horse_results_tmp)
        df_eval = engineer.add_jockey_features(df_eval)
        
        if not peds.empty:
            df_eval = engineer.add_pedigree_features(df_eval, peds)
            
        df_eval = engineer.create_target(df_eval, target_type='place')
        
        categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
        categorical_cols = [c for c in categorical_cols if c in df_eval.columns]
        df_eval = processor.encode_categorical(df_eval, categorical_cols)
        
        # Missing columns fill 0
        for col in model.feature_names:
            if col not in df_eval.columns:
                df_eval[col] = 0
                
        X_eval = df_eval[model.feature_names].copy()
        X_eval = X_eval.fillna(X_eval.median())
        
        if hasattr(processor, 'scaler') and processor.scaler:
            X_eval = processor.transform_scale(X_eval)
            
        y_eval = df_eval['target'].copy()
        
        # Prediction
        y_pred_proba = model.predict(X_eval)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Final Metrics
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_eval, y_pred_proba)
        except:
            auc = 0
            
        print(f"Results for {algo}: AUC={auc:.4f}, F1={f1:.4f}")
        
        results_list.append({
            'Model': algo,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc,
            'Time': training_time
        })
        
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model':<15} | {'AUC':<8} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Time':<8}")
    print("-" * 80)
    for res in results_list:
        print(f"{res['Model']:<15} | {res['AUC']:<8.4f} | {res['F1']:<8.4f} | {res['Precision']:<10.4f} | {res['Recall']:<8.4f} | {res['Time']:<8.1f}s")
    print("="*80)

if __name__ == "__main__":
    main()
