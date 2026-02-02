"""
2026年1月 フォーメーション戦略シミュレーション
- モデル: experiment_model_2025.pkl (Data Leakage Fixed)
- 戦略: formation
- 予算: 5000円
- 期間: 2026/01/01 - 2026/01/31
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

# パスの解決
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.constants import DATA_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE, RETURN_FILE, MODEL_DIR
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator

# 設定
MODEL_PATH = os.path.join(MODEL_DIR, 'experiment_model_2025.pkl')
RESULTS_2026_FILE = os.path.join(RAW_DATA_DIR, 'results_202601_fixed.pickle')
RETURN_2026_FILE = os.path.join(RAW_DATA_DIR, 'return_202601.pickle')

STRATEGY = 'formation'
BUDGET = 5000
START_DATE = '2026-01-01'
END_DATE = '2026-01-31'
REPORT_FILE = "january_2026_simulation_report.md"
DETAIL_FILE = "january_2026_simulation_details.md"

def load_resources():
    print("Loading resources...", flush=True)
    
    # モデル
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    model = HorseRaceModel()
    model.load(MODEL_PATH)
    
    # 払戻金データ (Main + 2026)
    print("Loading returns...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RETURN_FILE), 'rb') as f:
        returns_main = pickle.load(f)
        
    try:
        with open(RETURN_2026_FILE, 'rb') as f:
            returns_2026 = pickle.load(f)
    except FileNotFoundError:
        print("Warning: return_202601.pickle not found.")
        returns_2026 = pd.DataFrame()

    # Merge Returns
    if isinstance(returns_main, dict):
        # Convert dict to DF
        records = []
        for rid, data in returns_main.items():
            for row in data:
                records.append([rid] + row)
        returns_df = pd.DataFrame(records, columns=['race_id', 0, 1, 2, 3])
        returns_df.set_index('race_id', inplace=True)
    else:
        returns_df = returns_main
        
    if not returns_2026.empty:
        # returns_2026 is likely DF from Scraper
        # normalize format if needed? Scraper returns DF with race_id index?
        # Return.scrape returns DataFrame with index=race_id?
        # Let's check Return.scrape output.. usually it's DF with MultiIndex or index=race_id
        # Assuming typical concat works
        returns_df = pd.concat([returns_df, returns_2026])
        
    # Remove duplicates
    # returns_df = returns_df[~returns_df.index.duplicated(keep='last')] 
    # Actually duplicates might exist if same race_id (unlikely for different years)
    
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
    # returns pickle format: [type(0), winning_nums(1), payout(2), popularity(3)]?
    # Or Return.scrape format?
    # Return.scrape output df usually has columns [0, 1, 2, 'type'...] or similar depending on parsing.
    # The scraping.py says: pd.read_html... merged_df. 
    # Standard format often has columns like '券種', '馬番', '払戻', '人気'
    
    # Need to handle different return formats (Pickle vs Scraped DF)
    # Pickle (legacy) might be lists. Scraped DF has columns.
    
    # Helper to normalize row
    def normalize_row(row):
        # returns_df might be mixed.
        # If scraper output: columns might be ['0', '1', '2'] or Japanese ['券種'...]
        # Let's check row keys
        try:
            row_type = row.get(0, row.get('券種', ''))
            row_win = row.get(1, row.get('馬番', ''))
            row_pay = row.get(2, row.get('払戻', 0))
            return row_type, row_win, row_pay
        except: return None, None, 0
        
    hits_list = []
    if isinstance(hits, pd.DataFrame):
        for _, row in hits.iterrows():
            hits_list.append(normalize_row(row))
    elif isinstance(hits, pd.Series):
         hits_list.append(normalize_row(hits))
         
    for h_type, h_win, h_pay in hits_list:
        if h_type != bet_type: continue
        
        try:
            money_str = str(h_pay).replace(',', '').replace('円', '')
            pay = int(money_str)
            win_str = str(h_win).replace('→', '-')
            
            if '-' in win_str:
                win_nums = [int(x) for x in win_str.split('-')]
            else:
                win_nums = [int(win_str)]
                
            is_hit = False
            bet_horse_nums = set(rec.get('horse_numbers', []))
            
            # --- Strict Hit Verification ---
            if method == 'BOX':
                 if set(win_nums).issubset(bet_horse_nums):
                     is_hit = True
            elif method in ['FORMATION', '流し']:
                 structure = rec.get('formation', [])
                 if not structure:
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
                                 g1, g2 = set(structure[0]), set(structure[1])
                                 winners = set(win_nums)
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1 and (winners - axis_hit).issubset(g2):
                                     is_hit = True
                             elif len(structure) == 1:
                                 if set(win_nums).issubset(set(structure[0])): is_hit = True
                     elif bet_type in ['馬連', 'ワイド']:
                         if len(win_nums) == 2:
                             if len(structure) == 2:
                                 g1, g2 = set(structure[0]), set(structure[1])
                                 winners = set(win_nums)
                                 axis_hit = g1.intersection(winners)
                                 if len(axis_hit) >= 1 and (winners - axis_hit).issubset(g2):
                                     is_hit = True
                             elif len(structure) == 1:
                                 if set(win_nums).issubset(set(structure[0])): is_hit = True
            elif method == 'SINGLE':
                if bet_type == '単勝':
                    if win_nums[0] in bet_horse_nums: is_hit = True
                else:
                    if list(win_nums) == list(rec.get('horse_numbers', [])): is_hit = True

            if is_hit:
                unit = rec.get('unit_amount', 100) 
                payout += int(pay * (unit / 100))
                
        except Exception as e:
            continue
            
    return payout

def run_simulation():
    print("=== January 2026 Formation Strategy Simulation ===")
    
    # 1. Load Resources
    model, returns_df = load_resources()
    
    # 2. Load Data & Merge
    print("Loading race results...", flush=True)
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results_main = pickle.load(f)
        
    try:
        with open(RESULTS_2026_FILE, 'rb') as f:
            results_2026 = pickle.load(f)
        print(f"Loaded 2026 results: {len(results_2026)} rows")
    except FileNotFoundError:
        print("Error: results_202601.pickle not found. Run scraper first.")
        return

    # Merge
    if isinstance(results_main.index, pd.Index) and results_main.index.name == 'race_id':
         results_main = results_main.reset_index()
    elif 'race_id' not in results_main.columns and hasattr(results_main, 'index'):
        results_main['race_id'] = results_main.index.astype(str)
        
    if isinstance(results_2026.index, pd.Index):
        results_2026 = results_2026.reset_index(drop=True) # or keep index if it's race_id
    # Reset index to avoid issues, ensure race_id column exists
    if 'race_id' not in results_2026.columns:
        # Assuming index was race_id
        results_2026['race_id'] = results_2026.index.astype(str)
        
    results = pd.concat([results_main, results_2026], ignore_index=True)
    # results = results.drop_duplicates(subset=['race_id', '馬番'], keep='last') # Moved after processing
    
    # Ensure date column
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    else:
         # Normalize date format
         results['date'] = pd.to_datetime(results['date'], format='%Y年%m月%d日', errors='coerce')
         # Fill NaNs in date
         mask = results['date'].isnull()
         if mask.any():
              # Try deriving from race_id for missing dates
              # race_id likely in index or column? reset_index used, so likely in column if reset.
              if 'race_id' in results.columns:
                   results.loc[mask, 'date'] = pd.to_datetime(results.loc[mask, 'race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')

    print(f"Total Combined Results (Raw): {len(results)} rows")

    # Feature Engineering (On ALL data to ensure history is correct)
    print("Preprocessing & Feature Engineering...", flush=True)
    
    # WORKAROUND: Backup date column as it might be lost or corrupted in process_results
    # Map race_id -> date
    date_map = results.set_index('race_id')['date'].to_dict()
    
    processor = DataProcessor()
    df_proc = processor.process_results(results)
    
    # Restore date from backup
    print("Restoring date column from backup...")
    if 'race_id' not in df_proc.columns:
        df_proc['race_id'] = df_proc.index.astype(str)
        
    # Apply date map
    # Note: df_proc['race_id'] is a Series. map using dict.
    df_proc['date'] = df_proc['race_id'].map(date_map)
    
    # Convert to datetime to be sure
    df_proc['date'] = pd.to_datetime(df_proc['date'], errors='coerce')
    
    # Drop duplicates NOW, after column normalization
    if 'race_id' in df_proc.columns and '馬番' in df_proc.columns:
         df_proc = df_proc.drop_duplicates(subset=['race_id', '馬番'], keep='last')
    
    engineer = FeatureEngineer()
    
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    active_horses = df_proc['horse_id'].unique() if 'horse_id' in df_proc.columns else []
    hr_filtered = hr[hr.index.isin(active_horses)].copy() if not hr.empty else hr
    
    df_proc = engineer.add_horse_history_features(df_proc, hr_filtered)
    df_proc = engineer.add_course_suitability_features(df_proc, hr_filtered)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # DEBUG: Check date column
    print(f"Columns in df_proc: {len(df_proc.columns)}")
    if 'date' in df_proc.columns:
        print(f"Date Match Sample: {df_proc['date'].head()}")
        print(f"Date Range: {df_proc['date'].min()} - {df_proc['date'].max()}")
        print(f"Null Dates: {df_proc['date'].isnull().sum()}")
        
        # FIX: Recover Null dates from race_id
        if df_proc['date'].isnull().any():
             print("Attempting to recover Null dates in df_proc from race_id...")
             mask = df_proc['date'].isnull()
             if 'race_id' in df_proc.columns:
                 df_proc.loc[mask, 'date'] = pd.to_datetime(df_proc.loc[mask, 'race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
                 print(f"Null dates after recovery: {df_proc['date'].isnull().sum()}")
                 print(f"New Date Range: {df_proc['date'].min()} - {df_proc['date'].max()}")
             else:
                 print("Cannot recover dates: race_id column missing in df_proc")

    else:
        print("Date column MISSING in df_proc!")
        # Try to recover date from race_id
        if 'race_id' in df_proc.columns:
             df_proc['date'] = pd.to_datetime(df_proc['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
             print("Recovered date from race_id.")
    
    # Filter for January 2026
    mask = (df_proc['date'] >= pd.to_datetime(START_DATE)) & (df_proc['date'] <= pd.to_datetime(END_DATE))
    df_jan = df_proc[mask].copy()
    
    print(f"Target Races (Jan 2026): {len(df_jan['race_id'].unique())} races")
    
    if df_jan.empty:
        print("No races found for Jan 2026 after preprocessing.")
        return

    # Prepare Features
    feature_names = model.feature_names
    X = df_jan.copy()
    for col in feature_names:
        if col not in X.columns: X[col] = 0
    X = X[feature_names].fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Predict
    print("Predicting...", flush=True)
    probs = model.predict(X)
    df_jan['probability'] = probs
    df_jan['expected_value'] = df_jan['probability'] * df_jan.get('単勝', 0)
    
    # Simulation Loop
    unique_races = df_jan['race_id'].unique()
    details_log = []
    
    total_invest = 0
    total_return = 0
    hit_count = 0
    race_count = 0
    
    print("Simulating betting...", flush=True)
    for race_id in tqdm(unique_races):
        race_df = df_jan[df_jan['race_id'] == race_id].copy()
        
        preds = []
        for _, row in race_df.iterrows():
            preds.append({
                'horse_number': int(row.get('馬番', 0)),
                'horse_name': str(row.get('馬名', '')),
                'probability': float(row['probability']),
                'odds': float(row.get('単勝', 0)),
                'popularity': int(row.get('人気', 0)),
                'expected_value': float(row['expected_value'])
            })
        
        df_preds_alloc = pd.DataFrame(preds)
        recommendations = BettingAllocator.allocate_budget(
            df_preds_alloc, 
            budget=BUDGET, 
            strategy=STRATEGY
        )
        
        if not recommendations: continue
            
        race_invest = 0
        race_return = 0
        bets_str_list = []
        
        for rec in recommendations:
            cost = rec.get('total_amount', 0) 
            race_invest += cost
            pay = verify_hit(race_id, rec, returns_df)
            race_return += pay
            
            b_type = rec.get('bet_type', 'Unknown')
            method = rec.get('method', '')
            horses = rec.get('horse_numbers', [])
            
            if method == 'FORMATION' and 'formation' in rec:
                try:
                    fmt_str = "-".join([",".join(map(str, g)) for g in rec['formation']])
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
        if is_hit: hit_count += 1
            
        details_log.append({
            'race_id': race_id,
            'date': str(race_df.iloc[0]['date'])[:10],
            'bets': "<br>".join(bets_str_list),
            'invest': race_invest,
            'return': race_return,
            'balance': race_return - race_invest,
            'result': '🎯' if is_hit else '❌'
        })
            
    # Reporting
    recovery_rate = (total_return / total_invest * 100) if total_invest > 0 else 0
    hit_rate = (hit_count / race_count * 100) if race_count > 0 else 0
    
    report = f"""# 2026年1月 シミュレーション結果 (Formation)

- **モデル**: Experiment Model 2025 (Leakage Fixed - Trained on 2016-2024)
- **戦略**: {STRATEGY}
- **予算**: {BUDGET}円/レース
- **期間**: {START_DATE} - {END_DATE} (Scraped Data)
- **対象レース数**: {race_count} / {len(unique_races)}

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
    
    with open(DETAIL_FILE, 'w', encoding='utf-8') as f:
        f.write("# 2026年1月 レース別詳細ログ\n\n")
        f.write("| 日付 | レースID | 買い目 (金額) | 投資 | 回収 | 収支 | 結果 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for log in details_log:
            row = f"| {log['date']} | {log['race_id']} | {log['bets']} | {log['invest']:,} | {log['return']:,} | {log['balance']:,} | {log['result']} |\n"
            f.write(row)
    print(f"Detailed log saved to {DETAIL_FILE}")

if __name__ == "__main__":
    run_simulation()
