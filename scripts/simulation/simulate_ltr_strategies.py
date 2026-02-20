
import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import softmax # For robust softmax

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel
from modules.strategies.experimental import ExperimentalStrategies
from scripts.prediction.predict_20260214_ltr import RankingWrapper

def load_ltr_pipeline():
    """LTRモデルと関連リソースをロード"""
    print("Loading LTR model pipeline...")
    
    # 1. Load LTR Model (RankingWrapper logic)
    ltr_model_path = os.path.join(MODEL_DIR, 'standalone_ranking', 'ranking_model.pkl')
    if not os.path.exists(ltr_model_path):
        raise FileNotFoundError(f"LTR model not found at {ltr_model_path}")

    with open(ltr_model_path, 'rb') as f:
        data = pickle.load(f)
        
    model = RankingWrapper(data)
    print("LTR Model loaded.")
    
    # 2. Load Processor/Engineer (From historical_2010_2026 as in predict script)
    latest_model_dir = os.path.join(MODEL_DIR, 'historical_2010_2026')
    processor_path = os.path.join(latest_model_dir, 'processor_2026.pkl')
    engineer_path = os.path.join(latest_model_dir, 'engineer_2026.pkl')
    
    if not os.path.exists(processor_path):
        processor_path = os.path.join(latest_model_dir, 'processor.pkl')
    if not os.path.exists(engineer_path):
        engineer_path = os.path.join(latest_model_dir, 'engineer.pkl')
        
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    print("Processor loaded.")

    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)
    print("Engineer loaded.")
    
    return model, processor, engineer

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
    if 'original_race_id' in df_2025.columns:
        df_2025['race_id'] = df_2025['original_race_id'].astype(str)
    
    # Check if race_id exists now
    if 'race_id' not in df_2025.columns:
        # Check if 'id' or other column?
        # If not, try to construct it? No, critical.
        # But maybe 'date' + 'venue' + 'race_number'?
        print("Warning: 'race_id' column not found. Checking alternatives...")
        if 'race_id' in df.columns:
             # Maybe filtering lost it? No.
             pass
    else:
        # Ensure it's string
        df_2025['race_id'] = df_2025['race_id'].astype(str)

    # Alias columns
    if '馬番' in df_2025.columns and 'horse_number' not in df_2025.columns:
        df_2025['horse_number'] = df_2025['馬番']
    if '馬名' in df_2025.columns and 'horse_name' not in df_2025.columns:
        df_2025['horse_name'] = df_2025['馬名']

    print(f"Loaded {len(df_2025)} rows for 2025 simulation.")
    return df_2025

def simulate_strategies(df, model, processor, returns_data):
    """戦略シミュレーション実行"""
    
    if 'race_id' not in df.columns:
        print("Error: 'race_id' not found in DataFrame.")
        return {}
        
    results = {
        'dynamic_box_1000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0},
        'dynamic_box_3000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0},
        'value_hunter_5000': {'invest': 0, 'return': 0, 'hits': 0, 'races': 0}
    }
    
    race_ids = df['race_id'].unique()
    print(f"Simulating {len(race_ids)} races...")
    
    grouped = df.groupby('race_id')
    
    for rid, group in tqdm(grouped, total=len(race_ids)):
        # 1. Feature Preparation
        X = group.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X = X.fillna(0)
        
        if processor.scaler:
            try:
                 X = processor.transform_scale(X)
            except Exception:
                pass

        # 2. Prediction (LTR Scores)
        try:
            scores = model.predict(X)
        except Exception:
            continue
            
        # 3. Score Normalization (Softmax)
        probabilities = softmax(scores)

        # 4. Construct Prediction DataFrame
        cols_to_keep = ['horse_number', '単勝']
        if 'horse_name' in group.columns:
            cols_to_keep.append('horse_name')
            
        pred_df = group[cols_to_keep].copy()
        
        if 'horse_name' not in pred_df.columns:
            pred_df['horse_name'] = pred_df['horse_number'].astype(str)
            
        pred_df['probability'] = probabilities
        pred_df['odds'] = pd.to_numeric(pred_df['単勝'], errors='coerce').fillna(0)
        
        # 5. Get Odds Data
        race_returns = returns_data.get(rid, {})
        odds_data_mock = {}
        if 'tan' in race_returns:
             odds_data_mock['tan'] = {k: v/100.0 for k, v in race_returns['tan'].items()}

        # 6. Execute Strategies
        # Dynamic Box (1000)
        recs = ExperimentalStrategies.dynamic_box(pred_df, budget=1000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['dynamic_box_1000'])
        
        # Dynamic Box (3000)
        recs = ExperimentalStrategies.dynamic_box(pred_df, budget=3000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['dynamic_box_3000'])
        
        # Value Hunter (5000)
        recs = ExperimentalStrategies.value_hunter(pred_df, budget=5000, odds_data=odds_data_mock)
        calc_return(recs, race_returns, results['value_hunter_5000'])

    return results

def calc_return(recommendations, race_returns, stats):
    if not recommendations:
        return

    invest = 0
    payoff = 0
    
    for rec in recommendations:
        invest += rec['total_amount']
        
        btype = rec['bet_type']
        
        # Handle 'tan'/'fuku' (dict) vs others (list of dicts)
        map_key = {
            '単勝': 'tan', '複勝': 'fuku', '馬連': 'umaren',
            'ワイド': 'wide', '馬単': 'umatan', '3連複': 'sanrenpuku', '3連単': 'sanrentan'
        }
        key = map_key.get(btype)
        if not key or key not in race_returns: continue
        
        bet_coins = 0
        winning_data = race_returns[key]
        purchased = set(map(int, rec['horse_numbers']))
        
        if key in ['tan', 'fuku']:
            # winning_data is {horse_num: payout}
            for h in purchased:
                if h in winning_data:
                    # Unit amount check
                    bet_coins += winning_data[h] # Payout per 100 yen
        else:
            # winning_data is dict { (h1, h2...): payout }
            for winning_comb, payout in winning_data.items():
                winning_nums = set(map(int, winning_comb))
                hit = False
                
                if rec['method'] == 'BOX':
                     if winning_nums.issubset(purchased):
                         hit = True
                elif rec['method'] == 'SINGLE':
                    if winning_nums.issubset(purchased):
                        hit = True
                elif rec['method'] == '流し':
                    if 'formation' in rec:
                        axis = set(rec['formation'][0])
                        opponents = set(rec['formation'][1])
                        
                        # Nagashi Hit Logic
                        if winning_nums.issuperset(axis):
                            remaining = winning_nums - axis
                            if remaining.issubset(opponents):
                                hit = True
                    else:
                        pass
                
                if hit:
                    bet_coins += payout
        
        # Adjust payoff based on unit amount (assuming payout is for 100 yen)
        payoff += bet_coins * (rec['unit_amount'] / 100)

    if invest > 0:
        stats['races'] += 1
        stats['invest'] += invest
        stats['return'] += payoff
        if payoff > 0:
            stats['hits'] += 1

def load_returns():
    from modules.data_loader import load_payouts
    print("Loading returns via load_payouts(2025, 2025)...")
    try:
        returns_map = load_payouts(2025, 2025)
        print(f"Loaded returns for {len(returns_map)} races.")
        return returns_map
    except Exception as e:
        print(f"Error loading payouts: {e}")
        return {}

if __name__ == "__main__":
    # Load resources
    model, processor, engineer = load_ltr_pipeline()
    df_2025 = load_dataset()
    returns_map = load_returns()
    
    # Run Simulation
    results = simulate_strategies(df_2025, model, processor, returns_map)
    
    # Output Results
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Generate Report
    lines = []
    lines.append("\n## LTRモデル (LambdaRank) 検証結果 (2025)\n")
    lines.append("| 戦略 | 予算 | 投資総額 | 回収総額 | 回収率 | 的中率 |")
    lines.append("|---|---|---|---|---|---|")
    
    for key, stats in results.items():
        roi = (stats['return'] / stats['invest']) * 100 if stats['invest'] > 0 else 0
        hit_rate = (stats['hits'] / stats['races']) * 100 if stats['races'] > 0 else 0
        name_parts = key.split('_')
        budget = name_parts[-1]
        strat_name = "_".join(name_parts[:-1])
        
        lines.append(f"| {strat_name} | {budget} | {stats['invest']:,} | {int(stats['return']):,} | **{roi:.1f}%** | {hit_rate:.1f}% |")
    
    print("\n".join(lines))
    
    report_path = os.path.join(os.getcwd(), 'reports', 'strategy_comparison_2025.md')
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
