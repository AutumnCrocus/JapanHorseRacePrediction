
"""
Deploy 2026 Model Resources
- Extracts processor and engineer from dataset_2010_2025.pkl
- Saves them as processor_2026.pkl and engineer_2026.pkl in models/
"""
import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import MODEL_DIR

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset_2010_2025.pkl')

def deploy():
    print("=== Deploying 2026 Resources ===")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print("Loading dataset...")
    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)
        
    processor = dataset.get('processor')
    engineer = dataset.get('engineer')
    
    if not processor or not engineer:
        print("Processor or Engineer not found in dataset.")
        return
        
    # Save processor
    proc_path = os.path.join(MODEL_DIR, 'processor_2026.pkl')
    with open(proc_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"Saved processor to {proc_path}")
    
    # Save engineer
    eng_path = os.path.join(MODEL_DIR, 'engineer_2026.pkl')
    with open(eng_path, 'wb') as f:
        pickle.dump(engineer, f)
    print(f"Saved engineer to {eng_path}")
    
    print("Deployment complete.")

if __name__ == "__main__":
    deploy()
