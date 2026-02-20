
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

def simulate_deepfm():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        print("Data or Model file missing.")
        return

    # Load data
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Data load failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Import specialized libs AFTER pickle load
    import torch
    from tqdm import tqdm
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Add project root to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # Import DeepFM
    from modules.models.deepfm import DeepFM

    # Config
    feature_config = data['feature_config']
    sparse_info = feature_config['sparse']
    dense_info = feature_config['dense']
    sparse_names = [f['name'] for f in sparse_info]
    dense_names = [f['name'] for f in dense_info]
    
    # Test Data
    test_df = data['test_df'].copy()
    X_test_sparse = np.stack([data['test_model_input'][name] for name in sparse_names], axis=1)
    X_test_dense = np.stack([data['test_model_input'][name] for name in dense_names], axis=1) if dense_names else np.zeros((len(X_test_sparse), 0))
    
    # Tensor
    X_s = torch.LongTensor(X_test_sparse).to(DEVICE)
    X_d = torch.FloatTensor(X_test_dense).to(DEVICE)
    
    # Model
    model = DeepFM(feature_config, dnn_hidden_units=(256, 128), embedding_dim=8, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Predicting...")
    batch_size = 4096
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_s), batch_size):
            end = i + batch_size
            p = model(X_s[i:end], X_d[i:end])
            preds.append(p.cpu().numpy())
            
    preds = np.concatenate(preds).flatten()
    test_df['pred'] = preds
    
    # Simulation Logic
    # Verify performance
    print("Evaluating Performance...")
    
    # Group by race
    # We need odds to calculate return.
    # checking deepfm_data.pkl content: test_df has ['race_id', 'horse_id', 'target', 'date']
    # It assumes odds are NOT in test_df. We need to join with original dataset or load results?
    # Inspect columns.py showed 'odds' in dataset_2010_2025.pkl.
    # create_deepfm_data.py saved minimal columns.
    # To simulate return, we need ODDS.
    # I should load dataset_2010_2025.pkl again to get odds?
    # Or modify create_deepfm_data.py to include odds.
    # Re-running create_deepfm_data.py is expensive.
    # I'll load dataset_2010_2025.pkl here and merge.
    
    from modules.constants import PROCESSED_DATA_DIR
    import pickle
    
    ds_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    print(f"Loading original dataset from {ds_path} to get Odds...")
    with open(ds_path, 'rb') as f:
        ds = pickle.load(f)
        full_df = ds['data']
    
    # select needed cols: race_id, horse_id, odds, ranking(着順)
    # Note: test_df has 'target' (<=3). We need exact rank for payout calculation if not standard.
    # But usually 'target' is enough for accuracy check.
    # For return simulation, we need 'odds' and 'is_win'/'is_place'.
    
    # Rename map from create_deepfm_data
    # We need to ensure we merge correctly.
    # merge on race_id and horse_id.
    
    # full_df might have Japanese columns? check create_deepfm_data.py rename map.
    # it renamed them for internal df but saved 'test_df' from that internal df.
    # So test_df has race_id, horse_id.
    
    # full_df has 'race_id' (if reset_index done) or index.
    # inspect_columns showed it has 'race_id' (handled in create_deepfm_data).
    # Wait, create_deepfm_data modified LOCAL df. The pickle on disk is UNTOUCHED.
    # So full_df loaded from disk has 'race_id' in index likely.
    
    if 'race_id' not in full_df.columns:
        if full_df.index.name == 'race_id':
             full_df = full_df.reset_index()
        else:
             full_df = full_df.reset_index().rename(columns={'index': 'race_id'})

    # Columns of interest in full_df (Japanese?)
    # odds -> '単勝' or 'odds' (inspect said 'odds' was present? Yes, add_odds_features adds 'odds')
    # rank -> '着順'
    
    merge_cols = ['race_id', 'horse_id', 'odds', '着順', '単勝', '人気']
    # Filter only existing
    merge_cols = [c for c in merge_cols if c in full_df.columns]
    
    # Merge
    # ensure types match
    test_df['race_id'] = test_df['race_id'].astype(str)
    full_df['race_id'] = full_df['race_id'].astype(str)
    test_df['horse_id'] = test_df['horse_id'].astype(str)
    full_df['horse_id'] = full_df['horse_id'].astype(str)
    
    merged = pd.merge(test_df, full_df[merge_cols], on=['race_id', 'horse_id'], how='left')
    
    # Fill missing odds
    if 'odds' not in merged.columns and '単勝' in merged.columns:
        merged['odds'] = pd.to_numeric(merged['単勝'], errors='coerce')
    
    merged['odds'] = merged['odds'].fillna(1.0)
    
    # Simple Top-K Simulation
    # Top 1 Accuracy
    # Sort by pred desc
    
    races = merged.groupby('race_id')
    hits = 0
    total = 0
    return_amount = 0
    invest_amount = 0
    
    # Top 3 Box (Simple Strategy)
    box_hits = 0
    box_invest = 0
    box_return = 0
    
    print(f"Simulating {len(races)} races...")
    
    for rid, group in tqdm(races):
        group = group.sort_values('pred', ascending=False)
        
        # Top 1
        top1 = group.iloc[0]
        invest_amount += 100
        if top1['着順'] == 1:
            hits += 1
            return_amount += top1['odds'] * 100
            
        # Box 5
        if len(group) >= 5:
            box = group.iloc[:5]
            # 3renpuku box 5 = 10 points
            box_invest += 1000
            
            # Check hits (1,2,3 in box)
            top3 = group.sort_values('着順').iloc[:3]
            # If all top3 horses are in box
            top3_ids = set(top3['horse_id'])
            box_ids = set(box['horse_id'])
            
            if top3_ids.issubset(box_ids):
                box_hits += 1
                # Need 3renpuku payout. 
                # We don't have it in dataset...
                # Assume standard return for now or just count hits?
                # Simulation script usually loads payouts.
                pass

    print(f"--- Results (Single Win) ---")
    print(f"Races: {len(races)}")
    print(f"Hit Rate: {hits/len(races):.2%}")
    print(f"Return: {return_amount}")
    print(f"Invest: {invest_amount}")
    print(f"ROI: {return_amount/invest_amount*100:.2f}%" if invest_amount>0 else "ROI: 0%")
    
    # Save Report
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(f"# DeepFM Simulation Report\n\n")
        f.write(f"- Model: DeepFM (Cold Start Optimized)\n")
        f.write(f"- Test Data: 2025 ({len(races)} races)\n\n")
        f.write(f"## Single Win Strategy\n")
        f.write(f"- Hit Rate: {hits/len(races):.2%}\n")
        f.write(f"- ROI: {return_amount/invest_amount*100:.2f}%\n")

if __name__ == "__main__":
    simulate_deepfm()
