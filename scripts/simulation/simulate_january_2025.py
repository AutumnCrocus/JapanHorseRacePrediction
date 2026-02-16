"""
2025年1月 フォーメーション戦略シミュレーション
- モデル: experiment_model_2025.pkl (Data Leakage Fixed)
- 戦略: formation
- 予算: 5000円
- 期間: 2025/01/01 - 2025/01/31
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

# パスの解決
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import DATA_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE, RETURN_FILE, MODEL_DIR
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator

# 設定
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')
STRATEGY = 'formation'
BUDGET = 5000
START_DATE = '2025-01-01'
END_DATE = '2025-01-31'
REPORT_FILE = "january_simulation_report.md"

def load_resources():
    print("Loading resources...", flush=True)
    
    # モデル
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = HorseRaceModel()
    model.load(MODEL_PATH)
    
    # 払戻金データ
    print("Loading returns...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RETURN_FILE), 'rb') as f:
        returns = pickle.load(f)
        
    # Process returns for easier access
    if isinstance(returns, dict):
        # Transform dict to DataFrame if needed, or use as is. 
        # Usually returns.pickle is race_id -> list of lists.
        # convert to DF for easier lookup if possible, or keep as dict.
        # simulate_strategy_comparison uses pd.read_pickle which returns whatever object.
        # Let's assume it's a DF or Dict.
        pass
        
    # Helper to standardize returns to DF
    # If it is a dict: {race_id: [[type, combo, money, pop], ...]}
    if isinstance(returns, dict):
        records = []
        for rid, data in returns.items():
            for row in data:
                records.append([rid] + row)
        returns_df = pd.DataFrame(records, columns=['race_id', 0, 1, 2, 3])
        returns_df.set_index('race_id', inplace=True)
    else:
        returns_df = returns
        
    return model, returns_df

def verify_hit(race_id, rec, returns_df):
    """的中判定と払戻金計算 (Strict Logic)"""
    if race_id not in returns_df.index: return 0
    race_rets = returns_df.loc[race_id]
    
    payout = 0
    bet_type = rec.get('bet_type')
    method = rec.get('method', 'SINGLE')
    
    if isinstance(race_rets, pd.Series):
        hits = pd.DataFrame([race_rets])
    else:
        hits = race_rets
    
    # Filter by bet type
    hits = hits[hits[0] == bet_type]
    
    for _, h in hits.iterrows():
        try:
            money_str = str(h[2]).replace(',', '').replace('円', '')
            pay = int(money_str)
            win_str = str(h[1]).replace('→', '-')
            
            if '-' in win_str:
                win_nums = [int(x) for x in win_str.split('-')]
            else:
                win_nums = [int(win_str)]
                
            is_hit = False
            bet_horse_nums = set(rec.get('horse_numbers', []))
            
            # --- Strict Hit Verification ---
            
            if method == 'BOX':
                 # Box: All winning numbers must be in the selected horses
                 if set(win_nums).issubset(bet_horse_nums):
                     is_hit = True
                     
            elif method in ['FORMATION', '流し']:
                 structure = rec.get('formation', [])
                 if not structure:
                     # Fallback (Should typically not happen with correct allocator)
                     if set(win_nums).issubset(bet_horse_nums): is_hit = True 
                 else:
                     if bet_type == '3連単':
                         if len(win_nums) == 3 and len(structure) >= 3:
                             if win_nums[0] in structure[0] and \
                                win_nums[1] in structure[1] and \
                                win_nums[2] in structure[2]:
                                 is_hit = True
                     elif bet_type == '3連複':
                         if len(win_nums) == 3:
                             if len(structure) == 2:
                                 g1 = set(structure[0])
                                 g2 = set(structure[1])
                                 winners = set(win_nums)
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1:
                                     rem_winners = winners - axis_hit
                                     if rem_winners.issubset(g2):
                                         is_hit = True
                             elif len(structure) == 1: # Box treated as formation
                                 if set(win_nums).issubset(set(structure[0])):
                                     is_hit = True
                     elif bet_type in ['馬連', 'ワイド']:
                         if len(win_nums) == 2:
                             if len(structure) == 2:
                                 g1 = set(structure[0])
                                 g2 = set(structure[1])
                                 winners = set(win_nums)
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1:
                                     rem_winners = winners - axis_hit
                                     if rem_winners.issubset(g2):
                                         is_hit = True
                             elif len(structure) == 1:
                                 if set(win_nums).issubset(set(structure[0])):
                                     is_hit = True

            elif method == 'SINGLE':
                if bet_type == '単勝':
                    if win_nums[0] in bet_horse_nums: is_hit = True
                else:
                    if list(win_nums) == list(rec.get('horse_numbers', [])):
                         is_hit = True

            if is_hit:
                unit = rec.get('unit_amount', 100) 
                payout += int(pay * (unit / 100))
                
        except Exception as e:
            continue
            
    return payout

def run_simulation():
    print("=== January 2025 Formation Strategy Simulation ===")
    
    # 1. Load Resources
    model, returns_df = load_resources()
    
    # 2. Load Data (Optimized Loading)
    print("Loading race results...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
         results = results.reset_index()
    elif 'race_id' not in results.columns and hasattr(results, 'index'):
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        try: results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        except: pass
        
    # Filter for January 2025
    mask = (file_date := pd.to_datetime(results['date'])) >= pd.to_datetime(START_DATE)
    mask &= file_date <= pd.to_datetime(END_DATE)
    df_jan = results[mask].copy()
    
    print(f"Target Races (Jan 2025): {len(df_jan['race_id'].unique())} races")
    
    # Preprocessing
    print("Preprocessing...", flush=True)
    processor = DataProcessor()
    df_proc = processor.process_results(df_jan)
    
    engineer = FeatureEngineer()
    
    # Load support files for Feature Engineering
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    active_horses = df_proc['horse_id'].unique() if 'horse_id' in df_proc.columns else []
    hr_filtered = hr[hr.index.isin(active_horses)].copy() if not hr.empty else hr
    
    df_proc = engineer.add_horse_history_features(df_proc, hr_filtered)
    df_proc = engineer.add_course_suitability_features(df_proc, hr_filtered)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # Prepare Features
    feature_names = model.feature_names
    X = df_proc.copy()
    
    for col in feature_names:
        if col not in X.columns: X[col] = 0
            
    # Clean X
    X = X[feature_names].fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Predict
    print("Predicting...", flush=True)
    probs = model.predict(X)
    df_proc['probability'] = probs
    df_proc['expected_value'] = df_proc['probability'] * df_proc.get('単勝', 0)
    
    # Simulation Loop
    total_invest = 0
    total_return = 0
    hit_count = 0
    race_count = 0
    
    unique_races = df_proc['race_id'].unique()
    details_log = []
    
    print("Simulating betting...", flush=True)
    for race_id in tqdm(unique_races):
        race_df = df_proc[df_proc['race_id'] == race_id].copy()
        race_name = race_df.iloc[0].get('race_name', f"Race {race_id}") # Assuming race_name might be added or generic
        # race_name column might not exist or be empty, use race_id
        
        # Format for Allocator
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'horse_name': str(row.get('馬名', '')),
                'probability': float(row['probability']),
                'odds': float(row.get('単勝', 0)), # Use closing odds for simulation
                'popularity': int(row.get('人気', 0)),
                'expected_value': float(row['expected_value'])
            })
        
        # Allocate
        df_preds_alloc = pd.DataFrame(preds)
        recommendations = BettingAllocator.allocate_budget(
            df_preds_alloc, 
            budget=BUDGET, 
            strategy=STRATEGY
        )
        
        if not recommendations:
            continue
            
        race_invest = 0
        race_return = 0
        bets_str_list = []
        
        for rec in recommendations:
            cost = rec.get('total_amount', 0) 
            race_invest += cost
            
            pay = verify_hit(race_id, rec, returns_df)
            race_return += pay
            
            # Format bet string for log
            b_type = rec.get('bet_type', 'Unknown')
            method = rec.get('method', '')
            horses = rec.get('horse_numbers', [])
            
            if method == 'FORMATION' and 'formation' in rec:
                # Format formation: [1,2]-[3,4]-[5,6]
                fmt = rec['formation']
                # Convert list of lists to string
                # e.g. [[1], [2,3], [4,5]] -> "1-2,3-4,5"
                try:
                    fmt_str = "-".join([",".join(map(str, g)) for g in fmt])
                    bet_str = f"[{b_type} FMT] {fmt_str} ({cost}円)"
                except:
                    bet_str = f"[{b_type} FMT] {horses} ({cost}円)"
            elif method == 'BOX':
                bet_str = f"[{b_type} BOX] {horses} ({cost}円)"
            else:
                bet_str = f"[{b_type}] {horses} ({cost}円)"
                
            bets_str_list.append(bet_str)
            
        total_invest += race_invest
        total_return += race_return
        race_count += 1
        
        is_hit = race_return > 0
        if is_hit:
            hit_count += 1
            
        # Log Detail
        details_log.append({
            'race_id': race_id,
            'date': str(race_df.iloc[0]['date'])[:10],
            'bets': "<br>".join(bets_str_list),
            'invest': race_invest,
            'return': race_return,
            'balance': race_return - race_invest,
            'result': '🎯 EXACT MATCH' if is_hit and race_return > 0 else ('🎯 HIT' if is_hit else 'MISS') # Just HIT/MISS
        })
            
    # Reporting
    recovery_rate = (total_return / total_invest * 100) if total_invest > 0 else 0
    hit_rate = (hit_count / race_count * 100) if race_count > 0 else 0
    
    report = f"""# 2025年1月 シミュレーション結果 (Formation)

- **モデル**: Experiment Model 2025 (Leakage Fixed)
- **戦略**: {STRATEGY}
- **予算**: {BUDGET}円/レース
- **期間**: {START_DATE} - {END_DATE}
- **対象レース数**: {race_count} / {len(unique_races)} (投資対象のみ)

## 集計結果
| 項目 | 値 |
|---|---|
| 総投資額 | {total_invest:,} 円 |
| 総回収額 | {total_return:,} 円 |
| **回収率** | **{recovery_rate:.1f}%** |
| 的中数 | {hit_count} レース |
| 的中率 | {hit_rate:.1f}% |
| 収支 | {total_return - total_invest:,} 円 |

"""
    print("\n" + report)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {REPORT_FILE}")
    
    # Save Detail log
    DETAIL_FILE = "january_simulation_details.md"
    with open(DETAIL_FILE, 'w', encoding='utf-8') as f:
        f.write("# 2025年1月 レース別詳細ログ\n\n")
        f.write("| 日付 | レースID | 買い目 (金額) | 投資 | 回収 | 収支 | 結果 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for log in details_log:
            res_icon = "🎯" if log['return'] > 0 else "❌"
            row = f"| {log['date']} | {log['race_id']} | {log['bets']} | {log['invest']:,} | {log['return']:,} | {log['balance']:,} | {res_icon} |\n"
            f.write(row)
    print(f"Detailed log saved to {DETAIL_FILE}")

if __name__ == "__main__":
    run_simulation()
