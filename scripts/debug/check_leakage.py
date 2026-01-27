import pandas as pd
import matplotlib.pyplot as plt
import os

def check_score_distribution():
    file_path = 'rolling_prediction_details.csv'
    if not os.path.exists(file_path):
        file_path = 'data/rolling_prediction_details.csv'
        
    if not os.path.exists(file_path):
        print("File not found")
        return
        
    df = pd.read_csv(file_path)
    
    print(f"Total rows: {len(df)}")
    
    # Check Score Stats by Rank
    print("\n--- Score Stats by Rank (1-5) ---")
    vals = []
    for r in range(1, 6):
        scores = df[df['rank'] == r]['score']
        print(f"Rank {r}: Mean={scores.mean():.4f}, Std={scores.std():.4f}, Min={scores.min():.4f}, Max={scores.max():.4f}")
        vals.append(scores.mean())
        
    print(f"Rank >5: Mean={df[df['rank'] > 5]['score'].mean():.4f}")
    
    # Check AUC approximate
    # Rank 1 vs Others
    from sklearn.metrics import roc_auc_score
    try:
        y_true = (df['rank'] <= 3).astype(int) # Place prediction model
        score = df['score']
        auc = roc_auc_score(y_true, score)
        print(f"\nOverall Place AUC: {auc:.4f}")
    except:
        pass

if __name__ == "__main__":
    check_score_distribution()
