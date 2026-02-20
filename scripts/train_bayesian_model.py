
import os
import sys
import pickle
import numpy as np
import choix
from datetime import datetime

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import MODEL_DIR, PROCESSED_DATA_DIR

def train_bayesian_model():
    """choixを用いてPlackett-Luceモデルを学習する"""
    
    print("Loading Bayesian data...")
    data_path = os.path.join(os.path.dirname(PROCESSED_DATA_DIR), 'models', 'bayesian', 'bayesian_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    rankings = data['rankings']
    n_horses = data['n_horses']
    horse_to_id = data['horse_to_id']
    id_to_horse = data['id_to_horse']
    
    # I-LSR calculates dense transition matrix (N x N) -> MemoryError.
    # choix.opt_rankings uses Newton-CG (dense Hessian calculation in Top1Fcts) or BFGS (dense inv-Hessian).
    # Solution: Use L-BFGS-B manually using Top1Fcts.objective/gradient.
    
    print("optimizing log-likelihood (L-BFGS-B)...")
    from scipy.optimize import minimize
    from choix.opt import Top1Fcts
    
    # Create function object for rankings
    # alpha=0.1 (regularization)
    fcts = Top1Fcts.from_rankings(rankings, penalty=0.1)
    
    # Initial params
    x0 = np.zeros(n_horses)
    
    # Optimize using L-BFGS-B (Limited-memory BFGS) which avoids dense Hessian
    res = minimize(
        fcts.objective,
        x0,
        method='L-BFGS-B',
        jac=fcts.gradient,
        options={'maxiter': 100, 'disp': True}
    )
    
    if not res.success:
        print(f"Optimization warning: {res.message}")
        
    params = res.x

    print("Training complete.")

    print(f"Params shape: {params.shape}")
    print(f"Top 5 params: {np.sort(params)[-5:]}")
    
    # 結果保存
    model_data = {
        'params': params,
        'horse_to_id': horse_to_id,
        'id_to_horse': id_to_horse,
        'trained_at': datetime.now().isoformat()
    }
    
    output_dir = os.path.join(MODEL_DIR, 'bayesian')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'rating_model.pkl')
    
    with open(out_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"Saved Bayesian model to {out_path}")
    
    # 簡易確認: 最強馬トップ10
    top_indices = np.argsort(params)[::-1][:10]
    print("\nTop 10 Horses by Rating:")
    for idx in top_indices:
        rating = params[idx]
        name = id_to_horse.get(idx, f"Unknown_{idx}")
        print(f"{name}: {rating:.4f}")

if __name__ == "__main__":
    train_bayesian_model()
