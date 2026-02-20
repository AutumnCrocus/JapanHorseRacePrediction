import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.models.deepfm import DeepFM

DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/models/deepfm/deepfm_data.pkl')
MODEL_FILE = os.path.join(os.path.dirname(__file__), '../models/deepfm/deepfm_model.pth')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/processed/deepfm_scores.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4096

def save_deepfm_features():
    print(f"Device: {DEVICE}")
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    if 'train_df' not in data:
        print("Error: train_df metadata not found in pickle. Please regenerate data.")
        return

    # Helper to prepare tensors
    def prepare_tensors(model_input_dict, feature_config):
        sparse_names = [f['name'] for f in feature_config['sparse']]
        dense_names = [f['name'] for f in feature_config['dense']]
        
        X_sparse = np.stack([model_input_dict[name] for name in sparse_names], axis=1)
        X_dense = np.stack([model_input_dict[name] for name in dense_names], axis=1) if dense_names else np.zeros((len(X_sparse), 0))
        
        return torch.LongTensor(X_sparse), torch.FloatTensor(X_dense)

    print("Preparing tensors (Train)...")
    X_train_s, X_train_d = prepare_tensors(data['train_model_input'], data['feature_config'])
    print("Preparing tensors (Test)...")
    X_test_s, X_test_d = prepare_tensors(data['test_model_input'], data['feature_config'])

    # Combine
    X_all_s = torch.cat([X_train_s, X_test_s], dim=0)
    X_all_d = torch.cat([X_train_d, X_test_d], dim=0)
    
    # Meta data
    df_train = data['train_df']
    df_test = data['test_df']
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    if len(df_all) != len(X_all_s):
        print(f"Error: Length mismatch. Meta: {len(df_all)}, Tensors: {len(X_all_s)}")
        return

    # Load Model
    print("Loading model...")
    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found: {MODEL_FILE}")
        # Hint: Should we train first? 
        print("Please train the model first using scripts/train_deepfm.py")
        return

    feature_config = data['feature_config']
    model = DeepFM(feature_config, dnn_hidden_units=(256, 128), 
                   embedding_dim=8, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Predict
    print("Predicting...")
    preds = []
    dataset = torch.utils.data.TensorDataset(X_all_s, X_all_d)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for x_s, x_d in tqdm(loader):
            x_s, x_d = x_s.to(DEVICE), x_d.to(DEVICE)
            out = model(x_s, x_d)
            preds.append(out.cpu().numpy().flatten())

    preds = np.concatenate(preds)
    
    # Save
    print("Saving scores...")
    df_all['deepfm_score'] = preds
    
    # Keys for merging: race_id, horse_number
    # (horse_number_original is correct, but let's save as 'horse_number' for easier merge if possible, 
    # but upstream usually uses race_id + horse_number.
    # Check data_loader.py: load_results returns 'horse_number'.
    # Our df_all has 'horse_number_original' which comes from 'horse_number'.
    
    # Rename for consistency with data_loader expectations
    output_df = df_all[['race_id', 'horse_number_original', 'deepfm_score']].copy()
    output_df.rename(columns={'horse_number_original': 'horse_number'}, inplace=True)
    
    # Ensure race_id format matches (string) if needed
    # data_loader usually standardizes, but let's keep it as is from pickle
    
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"DeepFM scores saved to {OUTPUT_FILE}")
    print(output_df.head())

if __name__ == "__main__":
    save_deepfm_features()
