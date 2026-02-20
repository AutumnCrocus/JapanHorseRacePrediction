
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.constants import MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from modules.strategies.experimental import ExperimentalStrategies

def load_bayesian_model():
    """Bayesian Model (Rating Model) のロード"""
    model_path = os.path.join(MODEL_DIR, 'bayesian', 'rating_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    print(f"Loading Bayesian model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    return model_data

def load_dataset():
    """シミュレーション用データセットのロード (2025年分)"""
    file_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    if not os.path.exists(file_path):
         file_path = os.path.join(RAW_DATA_DIR, 'processed', 'dataset_2010_2025.pkl')

    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
        
    df = data_dict['data']
    
    # Date handling
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter 2025
    df_2025 = df[df['date'].dt.year == 2025].copy()
    
    # Race ID handling
    if 'original_race_id' in df_2025.columns and 'race_id' not in df_2025.columns:
        df_2025['race_id'] = df_2025['original_race_id'].astype(str)
        
    if 'race_id' not in df_2025.columns:
        raise KeyError("Dataset missing 'race_id'")
        
    # Alias columns (馬番, 馬名)
    if '馬番' in df_2025.columns and 'horse_number' not in df_2025.columns:
        df_2025['horse_number'] = df_2025['馬番']
    if '馬名' in df_2025.columns and 'horse_name' not in df_2025.columns:
        df_2025['horse_name'] = df_2025['馬名']
    if '着順' in df_2025.columns and 'rank' not in df_2025.columns:
        df_2025['rank'] = pd.to_numeric(df_2025['着順'], errors='coerce')

    print(f"Loaded {len(df_2025)} rows for 2025 simulation.")
    return df_2025

from modules.data_loader import load_payouts

def load_returns():
    """払い戻しデータのロード"""
    print("Loading returns data for 2025...")
    # Use standardized data loader
    returns_map = load_payouts(2025, 2025)
    return returns_map

def calc_return(recommendations, race_returns, stats):
    """回収率計算 (LTRシミュレーションから流用)"""
    if not recommendations:
        return

    invest = 0
    payoff = 0
    
    for rec in recommendations:
        invest += rec['total_amount']
        
        btype = rec['bet_type']
        
        # Handle 'tan'/'fuku' (dict) vs others (list of dicts/tuples)
        # 統一された形式: { 'tan': {horse_num: payout}, 'umaren': {(h1, h2): payout}, ... }
        map_key = {
            '単勝': 'tan', '複勝': 'fuku', '馬連': 'umaren',
            'ワイド': 'wide', '馬単': 'umatan', '3連複': 'sanrenpuku', '3連単': 'sanrentan'
        }
        key = map_key.get(btype)
        if not key or key not in race_returns: continue
        
        bet_coins = 0
        winning_data = race_returns[key]
        purchased = set(map(int, rec['horse_numbers']))
        
        # Check if winning_data is dict or list
        if isinstance(winning_data, list):
            # Convert list of dicts to flat dict for simple check
            # This handles old format if present
            pass 
            
        if key in ['tan', 'fuku']:
            # winning_data is {horse_num: payout}
            for h in purchased:
                if h in winning_data:
                    bet_coins += winning_data[h]
        else:
            # winning_data is dict { (h1, h2...): payout }
            for winning_comb, payout in winning_data.items():
                winning_nums = set(map(int, winning_comb))
                hit = False
                
                if rec['method'] == 'BOX':
                     if winning_nums.issubset(purchased):
                         hit = True
                elif rec['method'] == 'SINGLE': # Used for Tan/Fuku actually
                    if winning_nums.issubset(purchased):
                        hit = True
                        
                if hit:
                    bet_coins += payout
        
        payoff += bet_coins * (rec['unit_amount'] / 100)

    if invest > 0:
        stats['races'] += 1
        stats['invest'] += invest
        stats['return'] += payoff
        if payoff > 0:
            stats['hits'] += 1

def simulate_bayesian():
    # Load Model
    model_data = load_bayesian_model()
    params = model_data['params']
    horse_to_id = model_data['horse_to_id']
    id_to_horse = model_data['id_to_horse']
    
    # Param stats
    mean_param = np.mean(params)
    print(f"Model loaded. Mean rating: {mean_param:.4f}")
    
    # Load Data
    df = load_dataset()
    returns_data = load_returns()
    
    # Results container
    results = {
        'dynamic_box_1000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0},
        'dynamic_box_3000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0},
        'value_hunter_5000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0}
    }
    
    race_ids = df['race_id'].unique()
    print(f"Simulating {len(race_ids)} races...")
    
    grouped = df.groupby('race_id')
    
    for rid, group in tqdm(grouped, total=len(race_ids)):
        # 1. Get Ratings
        ratings = []
        for _, row in group.iterrows():
            h_name = str(row['horse_name'])
            h_id_key = row.get('horse_id', h_name) # Use name if id not in df
            
            # Map to model ID
            # Data creation script used horse_id column if present, else horse_name
            # Need to match the key used in training
            
            # Try to resolve ID
            mid = horse_to_id.get(h_id_key)
            if mid is None:
                mid = horse_to_id.get(h_name) # Fallback to name
                
            if mid is not None:
                val = params[mid]
            else:
                # Unknown horse: assign mean rating? or 0 (if log-space)?
                # Plackett-Luce params are strength gamma > 0.
                # If optimization was in log-space (usually is for numerical stability), we need exp.
                # choix returns parameters gamma (if ilsr) or theta (if optimize)?
                # choix.opt_rankings returns parameters theta s.t. gamma = exp(theta).
                # Wait, let's verify if choix returns gamma or theta (log-gamma).
                # choix documentation says: "params : numpy.ndarray. The (penalized) ML estimate of model parameters."
                # Plackett-Luce model P(i|S) = exp(theta_i) / sum exp(theta_j).
                # So params are likely log-strengths.
                val = mean_param # Average strength
            
            ratings.append(val)
            
        ratings = np.array(ratings)
        
        # 2. Probability Calculation (Softmax on scores)
        # Since params are likely log-strengths, softmax on params gives probabilities directly.
        probabilities = softmax(ratings)
        
        # 3. Construct Prediction DataFrame
        cols_to_keep = ['horse_number', '単勝']
        if 'horse_name' in group.columns:
            cols_to_keep.append('horse_name')
            
        pred_df = group[cols_to_keep].copy()
        
        if 'horse_name' not in pred_df.columns:
            pred_df['horse_name'] = pred_df['horse_number'].astype(str)
            
        pred_df['probability'] = probabilities
        pred_df['odds'] = pd.to_numeric(pred_df['単勝'], errors='coerce').fillna(0)
        
        # 4. Get Odds Data
        race_returns = returns_data.get(rid, {})
        odds_data_mock = {}
        if 'tan' in race_returns:
             odds_data_mock['tan'] = {k: v/100.0 for k, v in race_returns['tan'].items()} # Payout to Odds

        # 5. Execute Strategies
        # Dynamic Box (1000)
        recs = ExperimentalStrategies.dynamic_box(pred_df, budget=1000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['dynamic_box_1000'])
        
        # Dynamic Box (3000)
        recs = ExperimentalStrategies.dynamic_box(pred_df, budget=3000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['dynamic_box_3000'])
        
        # Value Hunter (5000)
        recs = ExperimentalStrategies.value_hunter(pred_df, budget=5000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['value_hunter_5000'])

    # Print Results
    print("\n--- Simulation Results (Bayesian Model) ---")
    for name, res in results.items():
        roi = (res['return'] / res['invest']) * 100 if res['invest'] > 0 else 0
        print(f"{name}: ROI {roi:.1f}% (Inv: {res['invest']}, Ret: {res['return']}, Hits: {res['hits']})")

if __name__ == "__main__":
    simulate_bayesian()
