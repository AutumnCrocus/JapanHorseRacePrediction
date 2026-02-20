
import pickle
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Constants
DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/models/deepfm/deepfm_data.pkl')
MODEL_FILE = os.path.join(os.path.dirname(__file__), '../models/deepfm/deepfm_model.pth')
OUTPUT_REPORT = os.path.join(os.path.dirname(__file__), '../reports/deepfm_simulation.md')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')

def simulate_deepfm():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        print("Data or Model file missing.")
        return

    # 1. Load Data
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        print("DeepFM Data loaded successfully.")
    except Exception as e:
        print(f"Data load failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load Evaluation Data (Odds) from CSV (Lighter)
    print("Loading evaluation data from CSV...")
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'prediction_2025_lgbm.csv')
    try:
        eval_df = pd.read_csv(csv_path)
        print("Evaluation CSV loaded.")
    except Exception as e:
        print(f"Evaluation CSV load failed: {e}")
        return

    # 3. Import Torch & DeepFM
    print("Importing PyTorch...")
    import torch
    from tqdm import tqdm
    
    # Path setup for DeepFM module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.models.deepfm import DeepFM

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    # 4. Prepare Data
    feature_config = data['feature_config']
    sparse_info = feature_config['sparse']
    dense_info = feature_config['dense']
    sparse_names = [f['name'] for f in sparse_info]
    dense_names = [f['name'] for f in dense_info]
    
    X_test_sparse = np.stack([data['test_model_input'][name] for name in sparse_names], axis=1)
    X_test_dense = np.stack([data['test_model_input'][name] for name in dense_names], axis=1) if dense_names else np.zeros((len(X_test_sparse), 0))
    
    X_s = torch.LongTensor(X_test_sparse).to(DEVICE)
    X_d = torch.FloatTensor(X_test_dense).to(DEVICE)
    
    print(f"DEBUG: X_s shape: {X_s.shape}, len: {len(X_s)}")
    print(f"DEBUG: X_d shape: {X_d.shape}")
    
    # 5. Load Model
    print("Loading Model...")
    model = DeepFM(feature_config, dnn_hidden_units=(256, 128), embedding_dim=8, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 6. Predict
    print("Predicting...")
    batch_size = 4096
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_s), batch_size):
            end = i + batch_size
            p = model(X_s[i:end], X_d[i:end])
            p_np = p.cpu().numpy()
            # Flatten predictions batch to 1D array
            preds.append(p_np.flatten())
            
    # Concatenate all batches
    preds = np.concatenate(preds)
    result_df = data['test_df'].copy()
    result_df['pred'] = preds
    
    # Debug: Print columns
    print("DEBUG: result_df columns:")
    print(result_df.columns.tolist())
    print("DEBUG: result_df index name:", result_df.index.name)
    print("DEBUG: result_df head:")
    print(result_df.head(2))
    
    # 7. Merge with Odds
    print("Merging evaluations...")
    # Normalize IDs for merge
    # CSV: race_id, horse_number, win_odds, actual_rank
    # result_df: race_id, horse_number (from pickle)
    
    # Check if race_id is in index
    if 'race_id' not in result_df.columns:
        result_df = result_df.reset_index()
        print("Reset index. New columns:", result_df.columns.tolist())
    
    result_df['race_id'] = result_df['race_id'].astype(str)
    
    # Check for horse_number
    h_col = None
    if 'horse_number_original' in result_df.columns:
        h_col = 'horse_number_original'
    elif 'horse_number' in result_df.columns:
        h_col = 'horse_number'
    elif '馬番' in result_df.columns:
        h_col = '馬番'
    elif 'horse_id' in result_df.columns:
        # Check if CSV has horse_id?
        # CSV `prediction_2025_lgbm.csv` header: race_id,horse_number,horse_name...
        # It does NOT have horse_id.
        # But maybe we can map?
        print("Warning: horse_number missing, but horse_id exists.")
    
    if h_col:
        result_df['horse_number'] = result_df[h_col].astype(int)
    else:
        print("Error: horse_number column missing in result_df")
        # Print all columns to be sure
        for c in result_df.columns:
            print(f"- {c}")
        return
    
    eval_df['race_id'] = eval_df['race_id'].astype(str)
    eval_df['horse_number'] = eval_df['horse_number'].astype(int)
    
    # Merge
    merged = pd.merge(result_df, eval_df[['race_id', 'horse_number', 'win_odds', 'actual_rank']], 
                      on=['race_id', 'horse_number'], how='inner')
    
    print(f"Merged samples: {len(merged)}")
    
    merged['odds'] = merged['win_odds'].fillna(1.0)
    merged['rank'] = pd.to_numeric(merged['actual_rank'], errors='coerce').fillna(99)

    # 8. Calculate Metrics
    print("Calculating ROI...")
    races = merged.groupby('race_id')
    
    single_hits = 0
    single_return = 0
    single_invest = 0
    
    box_hits = 0 # Top 3 in Box 5
    box_invest = 0
    
    for rid, group in tqdm(races):
        group = group.sort_values('pred', ascending=False)
        
        # Single Win Strategy (Top 1)
        top1 = group.iloc[0]
        single_invest += 100
        if top1['rank'] == 1:
            single_hits += 1
            single_return += top1['odds'] * 100
            
    roi = (single_return / single_invest * 100) if single_invest > 0 else 0
    hit_rate = (single_hits / len(races) * 100) if len(races) > 0 else 0
    
    print(f"\n=== DeepFM Simulation Results (2025) ===")
    print(f"Races: {len(races)}")
    print(f"Strategy: Single Win (Top 1)")
    print(f"Hit Rate: {hit_rate:.2f}%")
    print(f"Return: {single_return} / {single_invest}")
    print(f"ROI: {roi:.2f}%")
    
    # Save Report
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("# DeepFM Simulation Report (2025)\n\n")
        f.write(f"- Races: {len(races)}\n")
        f.write(f"- Strategy: Single Win (Top 1 Score)\n")
        f.write(f"- Hit Rate: {hit_rate:.2f}%\n")
        f.write(f"- Return: {single_return} / {single_invest}\n")
        f.write(f"- ROI: {roi:.2f}%\n")

if __name__ == "__main__":
    simulate_deepfm()
