
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.models.deepfm import DeepFM

DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/models/deepfm/deepfm_data.pkl')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/deepfm')
MODEL_FILE = os.path.join(MODEL_DIR, 'deepfm_model.pth')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3

def train_deepfm():
    print(f"Device: {DEVICE}")
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    # config
    feature_config = data['feature_config']
    sparse_info = feature_config['sparse']
    dense_info = feature_config['dense']
    
    # sparse inputs order
    sparse_names = [f['name'] for f in sparse_info]
    dense_names = [f['name'] for f in dense_info]
    
    # Prepare Tensors
    print("Preparing tensors...")
    X_train_sparse = np.stack([data['train_model_input'][name] for name in sparse_names], axis=1)
    X_train_dense = np.stack([data['train_model_input'][name] for name in dense_names], axis=1) if dense_names else np.zeros((len(X_train_sparse), 0))
    y_train = data['train_target']
    
    X_test_sparse = np.stack([data['test_model_input'][name] for name in sparse_names], axis=1)
    X_test_dense = np.stack([data['test_model_input'][name] for name in dense_names], axis=1) if dense_names else np.zeros((len(X_test_sparse), 0))
    y_test = data['test_target']

    # Convert to Tensor
    X_train_sparse = torch.LongTensor(X_train_sparse)
    X_train_dense = torch.FloatTensor(X_train_dense)
    y_train = torch.FloatTensor(y_train)
    
    X_test_sparse = torch.LongTensor(X_test_sparse)
    X_test_dense = torch.FloatTensor(X_test_dense)
    y_test = torch.FloatTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_sparse, X_train_dense, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = DeepFM(feature_config, dnn_hidden_units=(256, 128), 
                   embedding_dim=8, device=DEVICE)
    model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_s, x_d, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x_s, x_d, y = x_s.to(DEVICE), x_d.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x_s, x_d)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation (Test set 2025)
        model.eval()
        with torch.no_grad():
            x_s_test, x_d_test, y_test_dev = X_test_sparse.to(DEVICE), X_test_dense.to(DEVICE), y_test.to(DEVICE)
            pred_test = model(x_s_test, x_d_test)
            val_loss = criterion(pred_test, y_test_dev).item()
            
            # Metrics
            y_true = y_test.numpy()
            y_pred = pred_test.cpu().numpy()
            auc = roc_auc_score(y_true, y_pred)
            
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, AUC={auc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break
                
    print("Training finished.")

if __name__ == "__main__":
    train_deepfm()
