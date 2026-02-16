
import os
import sys
import pandas as pd
import numpy as np
import pickle
import traceback
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, MODEL_DIR, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, EnsembleModel, RacePredictor
from modules.betting_allocator import BettingAllocator
from modules.strategy import BettingStrategy

# --- Strategy Classes ---
class Strategy:
    def name(self): return "Base"
    def generate_bets(self, df_preds, budget): return []

class CurrentStrategy(Strategy):
    def name(self): return "Current(Baseline)"
    def generate_bets(self, df_preds, budget):
        return BettingAllocator.allocate_budget(df_preds, budget, odds_data=None)

class BoxExpansionStrategy(Strategy):
    def name(self): return "HighBudget_MoreHorses"
    def generate_bets(self, df_preds, budget):
         # Copy of Current but with logic to PRIORITIZE Adding Horses if Budget >= 5000
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top = df_sorted['horse_number'].tolist()
        if not top: return []
        
        recs = []
        rem = budget
        
        if budget >= 5000:
            # 3RenTan 4-Box (2400)
            if len(top) >= 4:
                recs.append(self._box('3連単', top[:4], 2400))
                rem -= 2400
            
            # Expansion: 3RenPuku 6-Box (2000) instead of 5-Box(1000)
            if len(top) >= 6 and rem >= 2000:
                recs.append(self._box('3連複', top[:6], 2000))
                rem -= 2000
            elif len(top) >= 5 and rem >= 1000:
                recs.append(self._box('3連複', top[:5], 1000))
                rem -= 1000
                
             # Umaren 5-Box
            if len(top) >= 5 and rem >= 1000:
                recs.append(self._box('馬連', top[:5], 1000))
                rem -= 1000
        else:
            return BettingAllocator.allocate_budget(df_preds, budget)

        # Remainder to Win
        if rem >= 100:
            recs.append(self._win(top[0], (rem//100)*100))
            
        return self._format(recs)

    def _box(self, type, horses, amount):
        return {'type': type, 'method': 'BOX', 'horses': horses, 'amount': amount}
    def _win(self, horse, amount):
        return {'type': '単勝', 'method': 'SINGLE', 'horses': [horse], 'amount': amount}
    
    def _format(self, recs):
        # Translate to Allocator format structure
        out = []
        for r in recs:
            out.append({
                'bet_type': r['type'],
                'method': r['method'],
                'horse_numbers': r['horses'],
                'total_amount': r['amount'],
                'combination': 'BOX' if r['method']=='BOX' else str(r['horses'][0])
            })
        return out

class UnitIncreaseStrategy(Strategy):
    def name(self): return "HighBudget_UnitIncrease"
    def generate_bets(self, df_preds, budget):
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top = df_sorted['horse_number'].tolist()
        
        if budget >= 5000 and len(top)>=4:
             # Allocate max to 3RenTan Box
             recs = []
             # Base cost 2400 (24pts)
             # Multiplier
             unit = (budget // 2400) * 100
             if unit < 100: unit = 100
             cost = 24 * unit
             
             recs.append({
                 'bet_type': '3連単', 'method': 'BOX', 'horse_numbers': top[:4],
                 'total_amount': cost, 'unit_amount': unit, 'combination': 'BOX'
             })
             rem = budget - cost
             if rem >= 100:
                 recs.append({'bet_type': '単勝', 'method': 'SINGLE', 'horse_numbers': [top[0]], 'total_amount': (rem//100)*100, 'combination': str(top[0])})
             return recs
        else:
             return BettingAllocator.allocate_budget(df_preds, budget)

class NagashiStrategy(Strategy):
    def name(self): return "Nagashi_1Axis_Fixed"
    def generate_bets(self, df_preds, budget):
        top = df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()
        if len(top)<3: return []
        
        # 3RenTan 1st Fixed: 1 -> 2,3,4,5,6 -> 2,3,4,5,6
        # Pts = K * (K-1) where K is number of partners
        
        recs = []
        
        # Calculate max partners
        best_k = 0
        for k in range(2, 10): # 2 to 9 partners
            pts = k * (k-1)
            if pts * 100 <= budget:
                best_k = k
            else: break
            
        if best_k >= 2:
            partners = top[1:best_k+1]
            recs.append({
                'bet_type': '3連単', 'method': 'NAGASHI_AXIS_1', # Custom key for evaluator
                'axis': top[0], 'partners': partners,
                'horse_numbers': [top[0]] + partners,
                'total_amount': best_k*(best_k-1)*100,
                'combination': 'NAGASHI'
            })
            
            rem = budget - (best_k*(best_k-1)*100)
            if rem >= 100:
                recs.append({'bet_type': '単勝', 'method': 'SINGLE', 'horse_numbers': [top[0]], 'total_amount': (rem//100)*100, 'combination': str(top[0])})
        else:
             return BettingAllocator.allocate_budget(df_preds, budget)
        return recs

class FormationStrategy(Strategy):
    def name(self): return "Formation_12-14-16"
    def generate_bets(self, df_preds, budget):
        top = df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()
        if len(top)<6: return BettingAllocator.allocate_budget(df_preds, budget)
        
        # Formation: 1st(1-2), 2nd(1-4), 3rd(1-6)
        g1 = top[:2]
        g2 = top[:4]
        g3 = top[:6]
        
        def calc_pts(r1, r2, r3):
            c = 0
            for i in r1:
                for j in r2:
                    if i==j: continue
                    for k in r3:
                        if k==i or k==j: continue
                        c+=1
            return c
            
        pts = calc_pts(g1, g2, g3)
        cost = pts * 100
        
        if cost > budget:
            # Shrink 3rd row to 4
            g3 = top[:4]
            pts = calc_pts(g1, g2, g3)
            cost = pts * 100
            
        if cost <= budget:
            return [{
                'bet_type': '3連単', 'method': 'FORMATION',
                'formation': [g1, g2, g3],
                'horse_numbers': list(set(g1+g2+g3)),
                'total_amount': cost,
                'combination': 'FORMATION'
            }]
            
        return BettingAllocator.allocate_budget(df_preds, budget)

# --- Simulator Logic ---

def load_resources():
    print("Loading resources...")
    # Model
    if os.path.exists(os.path.join(MODEL_DIR, 'production_model.pkl')):
        model = HorseRaceModel()
        model.load(os.path.join(MODEL_DIR, 'production_model.pkl'))
    else:
        model = HorseRaceModel()
        model.load()
        
    with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    
    predictor = RacePredictor(model, processor, engineer)
    
    # Historical Data for Features
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    # 2025 Data
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    if isinstance(results.index, pd.Index):
        results['race_id'] = results.index.astype(str)
        
    # Filter hr for speed
    # Get all horse_ids in 2025 results
    active_horses = results['horse_id'].unique()
    print(f"Filtering history for {len(active_horses)} active horses from {len(hr)} records...")
    # hr index is horse_id
    hr = hr[hr.index.isin(active_horses)].copy()
    print(f"History filtered to {len(hr)} records.")
    
    # Returns
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f: returns = pickle.load(f)

    return predictor, hr, peds, results, returns

def process_race_mock(race_id, race_df_rows, predictor, hr, peds):
    # Construct DF similar to Shutuba.scrape
    # Columns needed: '馬番', '枠番', '馬名', '性齢', '斤量', '騎手', '調教師', '馬主', 'race_type', 'course_len', 'weather', 'ground_state', 'date'
    
    # race_df_rows is the slice of results.pickle for this race
    # Map columns
    
    df = race_df_rows.copy()
    
    # Rename map based on scraping.py vs results columns
    # results.pickle columns: ['着順', '枠 番', '馬 番', '馬名', '性齢', '斤量', '騎手', 'タイム', '着差', '通過', '上り', '単勝', '人気', '馬体重', '調教師', 'horse_id', 'jockey_id', 'trainer_id', 'race_type', 'course_len', 'weather', 'ground_state', 'date']
    
    rename_map = {
        '枠 番': '枠番',
        '馬 番': '馬番',
    }
    df = df.rename(columns=rename_map)
    
    # process_results expects specific columns
    # We rely on predictor.processor to handle it?
    # processor often processes raw scraped DF.
    
    # Fill missing expected columns if any (Shutuba has '開催', 'R' etc but maybe not critical for ML if not used)
    
    # IMPORTANT: '単勝' in results is the Result Odds.
    # We use this as the "Current Odds" for betting logic proxy.
    # Fix Date
    if 'date' in df.columns:
        try:
            # results.pickle has 'YYYY年MM月DD日'
            df['date'] = pd.to_datetime(df['date'], format='%Y年%m月%d日')
        except:
            # Fallback if different format or already datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # Force default date if missing to avoid Feature Engineering crash
        df['date'] = pd.Timestamp('2025-01-01')

    # Fill NaT with default
    df['date'] = df['date'].fillna(pd.Timestamp('2025-01-01'))
    
    # Clean types (ensure numeric)
    df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce')
    df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce')
    
    # Predict
    # We need to construct the pipeline manually or trust predict_race
    
    try:
        # Preprocess
        df_proc = predictor.processor.process_results(df)
        
        # OPTIMIZATION: Filter History for this race only
        if 'horse_id' in df.columns:
            race_horse_ids = df['horse_id'].unique()
            # hr index is horse_id
            race_hr = hr[hr.index.isin(race_horse_ids)]
        else:
            race_hr = hr
            
        # Add Features
        df_proc = predictor.engineer.add_horse_history_features(df_proc, race_hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, race_hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc) # Expects 単勝
        
        # Predict
        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            X[c] = df_proc[c] if c in df_proc.columns else 0
            
        # Categorical handling - Important for LightGBM
        try:
            debug_info = predictor.model.debug_info()
            model_cats = debug_info.get('pandas_categorical', [])
            if len(model_cats) >= 2:
                 if '枠番' in X.columns:
                     # Ensure it's treated as categorical with specific categories
                     cat_type = pd.CategoricalDtype(categories=model_cats[0], ordered=False)
                     X['枠番'] = X['枠番'].astype(cat_type)
                 if '馬番' in X.columns:
                     cat_type = pd.CategoricalDtype(categories=model_cats[1], ordered=False)
                     X['馬番'] = X['馬番'].astype(cat_type)
        except Exception as e:
            pass # print(f"Warning in categorical conversion: {e}")

        # Fix Categories / Objects
        for c in X.columns:
            if X[c].dtype == 'object':
                try: X[c] = pd.to_numeric(X[c])
                except: 
                     # print(f"Warning: Filling {c} with 0")
                     X[c] = 0
        
        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        # Odds might be in '単勝' or 'odds'
        if '単勝' in df_res.columns:
            df_res['odds'] = pd.to_numeric(df_res['単勝'], errors='coerce').fillna(0)
        elif 'odds' in df_res.columns:
            df_res['odds'] = df_res['odds']
        else:
             df_res['odds'] = 0
             
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
    except Exception as e:
        print(f"Predict Error: {e}")
        # traceback.print_exc()
        return None

def verify_hit(race_id, rec, returns_df):
    if race_id not in returns_df.index: return 0
    race_rets = returns_df.loc[race_id]
    
    payout = 0
    btype = rec['bet_type']
    method = rec.get('method', 'SINGLE')
    
    # Get winning numbers for this type
    hits = race_rets[race_rets[0] == btype]
    
    for _, h in hits.iterrows():
        try:
            val_str = str(h[2]).replace(',','').replace('円','')
            pay = int(val_str)
            w_str = str(h[1])
            w_str = w_str.replace('→', '-')
            if '-' in w_str: w_nums = [int(x) for x in w_str.split('-')]
            else: w_nums = [int(w_str)]
            
            is_hit = False
            
            if method == 'SINGLE':
                if w_nums[0] in rec['horse_numbers']: is_hit = True
            elif method == 'BOX':
                if set(w_nums).issubset(set(rec['horse_numbers'])): is_hit = True
            elif method == 'NAGASHI_AXIS_1':
                # Axis must be in w_nums. Others must be in partners.
                axis = rec['axis']
                partners = rec['partners']
                # AND we need to check if the combination matches logic
                # For 3RenTan Nagashi 1->P->P:
                # w_nums[0] MUST be axis. (Since it's 1st fixed)
                if w_nums[0] == axis and set(w_nums[1:]).issubset(set(partners)): is_hit = True
                
            elif method == 'FORMATION':
                # Ordered check for 3RenTan
                form = rec['formation']
                if btype == '3連単':
                    if len(w_nums)==3:
                        if w_nums[0] in form[0] and w_nums[1] in form[1] and w_nums[2] in form[2]:
                            is_hit = True
                            
            if is_hit:
                unit = rec.get('unit_amount', 100)
                payout += (pay * (unit/100))
        except: continue
        
    return payout

def main():
    print("Initializing Simulation...")
    predictor, hr, peds, results, returns = load_resources()
    
    strategies = [
        CurrentStrategy(),
        BoxExpansionStrategy(),
        UnitIncreaseStrategy(),
        NagashiStrategy(),
        FormationStrategy()
    ]
    
    # Filter for 2025
    df_2025 = results[results['race_id'].str.startswith('2025')]
    race_ids = df_2025['race_id'].unique()
    
    # 50 races for Rigorous Backtest (Optimized)
    import random
    random.seed(42)
    sample_rids = random.sample(list(race_ids), min(len(race_ids), 50))
    
    print(f"Simulating {len(sample_rids)} races with Budget 10,000 JPY...")
    
    metrics = {s.name(): {'cost':0, 'return':0, 'hits':0, 'races':0} for s in strategies}
    
    for rid in sample_rids:
        race_rows = df_2025[df_2025['race_id'] == rid]
        
        # Predict
        preds = process_race_mock(rid, race_rows, predictor, hr, peds)
        if preds is None or preds.empty:
             print("Preds empty/failed")
             continue
        else:
             pass # print(f"Preds OK: {len(preds)}")
        
        for strat in strategies:
            try:
                bets = strat.generate_bets(preds, budget=10000)
                if not bets:
                     # print(f"No bets for {strat.name()}")
                     pass
                
                cost = 0
                ret = 0
                hit = 0
                
                for b in bets:
                    c = b.get('total_amount', 0)
                    cost += c
                    p = verify_hit(rid, b, returns)
                    ret += p
                    if p > 0: hit = 1
                
                metrics[strat.name()]['cost'] += cost
                metrics[strat.name()]['return'] += ret
                metrics[strat.name()]['hits'] += hit
                metrics[strat.name()]['races'] += 1
                
            except Exception as e:
                # print(f"Strat Error {strat.name()}: {e}")
                pass
                
    # Report
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write("\n\n### Simulation Results (2025 Sample)\n")
        f.write(f"Races: {metrics[strategies[0].name()]['races']} / {len(sample_rids)}\n")
        f.write("| Strategy | Cost | Return | Recovery Rate | Hit Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        for s in strategies:
            m = metrics[s.name()]
            recov = (m['return'] / m['cost']) * 100 if m['cost']>0 else 0
            hrate = (m['hits'] / m['races']) * 100 if m['races']>0 else 0
            f.write(f"| {s.name()} | {m['cost']:,} | {int(m['return']):,} | {recov:.1f}% | {hrate:.1f}% |\n")
            
    print("Report written to report.md")

if __name__ == "__main__":
    main()
