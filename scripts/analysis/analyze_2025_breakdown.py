import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import re

# パスの解決
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import DATA_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE, RETURN_FILE, MODEL_DIR, PLACE_DICT
from modules.training import HorseRaceModel
from modules.preprocessing import DataProcessor, FeatureEngineer
from modules.betting_allocator import BettingAllocator

# 設定
VALIDATION_DIR = os.path.join(MODEL_DIR, "validation_2024")
DATE_MAP_FILE = os.path.join(DATA_DIR, "date_map_2025.pickle")
REPORT_FILE = "breakdown_2025_report.md"

# 逆引き用 (開催名 -> ID)
PLACE_NAME_TO_ID = {v: k for k, v in PLACE_DICT.items()}

def create_race_info_map():
    print("Building race info map using date_map and horse_results...", flush=True)
    if not os.path.exists(DATE_MAP_FILE):
        print(f"Error: {DATE_MAP_FILE} not found. Run scripts/scraping/build_date_map_2025.py first.")
        return {}
    
    with open(DATE_MAP_FILE, 'rb') as f:
        rid_map = pickle.load(f) # Key: race_id, Value: YYYY-MM-DD
    
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    
    # カラム名の正規化
    hr.columns = hr.columns.str.replace(' ', '')
    
    # horse_results から 開催日/場所/R -> (馬場, 距離) のマップを作成
    # 日付形式を YYYY-MM-DD に統一
    hr_map = {}
    print("  - Parsing horse_results for surface/distance...")
    
    # Use 'レース名' logic as confirmed to work
    if 'レース名' in hr.columns:
        for _, row in hr.iterrows():
            try:
                # Normalize text (Full-width to Half-width, etc)
                import unicodedata
                r_name = unicodedata.normalize('NFKC', str(row['レース名']))
                
                # "YY/MM/DD 会場名 ..."
                match = re.search(r'(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\d+)R', r_name)
                if not match: continue
                
                date_short, v_info, r_num = match.groups()
                year_prefix = "20" if int(date_short[:2]) < 50 else "19"
                date_full = f"{year_prefix}{date_short.replace('/', '-')}"
                
                # 会場名抽出 (例: "1中山5" -> "中山", or just "中山")
                # Also handle "高知" etc if processing local, but here assume JRA logic mostly
                v_match = re.search(r'([^\d]+)', v_info)
                if not v_match: continue
                v_name = v_match.group(1).replace('回', '').replace('日', '').strip() 
                # Note: v_info in r_name usually "高知" or "中山"
                
                key = (date_full, v_name, int(r_num))
                if key not in hr_map:
                    dist_raw = str(row['距離'])
                    dist_match = re.search(r'(\D+)(\d+)', dist_raw)
                    if dist_match:
                        hr_map[key] = {
                            'surface': dist_match.group(1),
                            'distance': int(dist_match.group(2)),
                            'entrants': int(row.get('頭数', 0)) if '頭数' in row else 0,
                            'name': r_name
                        }
            except: continue
            
    # Map race_id -> metadata
    race_meta = {}
    print("  - Mapping race_ids to metadata...")
    for race_id, date_str in rid_map.items():
        # Parse race_id for venue code -> venue name
        v_id = race_id[4:6]
        v_name = PLACE_DICT.get(v_id)
        if not v_name: continue
        
        # Parse race_id for R
        try:
            r_num = int(race_id[-2:])
        except: continue
        
        key = (date_str, v_name, r_num)
        
        if key in hr_map:
            race_meta[race_id] = hr_map[key]
        else:
            # Fallback or Log
            # print(f"Meta missing for {race_id} ({key})")
            race_meta[race_id] = {'surface': '不明', 'distance': 0, 'entrants': 0, 'name': ''}
                
    print(f"  - Mapped {len(race_meta)} races with metadata.")
    return race_meta

def get_age_cat(name):
    if '2歳' in name: return '2歳'
    if '3歳' in name: return '3歳'
    if '4歳以上' in name or '3歳以上' in name: return '4歳以上'
    # Implicit for named races? Assume 4yo+ if unmentioned
    return '4歳以上'

def get_class_cat(name):
    # Order matters
    if 'G1' in name or 'GI' in name: return '重賞'
    if 'G2' in name or 'GII' in name: return '重賞'
    if 'G3' in name or 'GIII' in name: return '重賞'
    
    if '新馬' in name: return '新馬'
    if '未勝利' in name: return '未勝利'
    
    if '1勝' in name or '500万' in name: return '1勝クラス'
    if '2勝' in name or '1000万' in name: return '2勝クラス'
    if '3勝' in name or '1600万' in name: return '3勝クラス'
    
    # OP list or pattern
    if 'OP' in name or 'オープン' in name or '(L)' in name: return 'オープン/その他'
    # If named race but no class specified, it's usually Open or Graded (already caught) or some special condition
    return 'オープン/その他'


def get_entrants_cat(count):
    if count == 0: return '不明'
    if count <= 9: return '少頭数 (~9)'
    if count <= 12: return '中頭数 (10~12)'
    return '多頭数 (13~)'

def load_resources():
    print("Loading resources...", flush=True)
    model = HorseRaceModel()
    model.load(os.path.join(VALIDATION_DIR, 'model.pkl'))
    with open(os.path.join(VALIDATION_DIR, 'processor.pkl'), 'rb') as f: processor = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'engineer.pkl'), 'rb') as f: engineer = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'bias_map.pkl'), 'rb') as f: bias_map = pickle.load(f)
    with open(os.path.join(VALIDATION_DIR, 'jockey_stats.pkl'), 'rb') as f: jockey_stats = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, RETURN_FILE), 'rb') as f: returns = pickle.load(f)
    
    if isinstance(returns, dict):
        records = []
        for rid, data in returns.items():
            for row in data: records.append([rid] + row)
        returns_df = pd.DataFrame(records, columns=['race_id', 0, 1, 2, 3]).set_index('race_id')
    else: 
        returns_df = returns
        # MultiIndexのLevel 0を文字列にする
        if isinstance(returns_df.index, pd.MultiIndex):
            returns_df.index = returns_df.index.set_levels(returns_df.index.levels[0].astype(str), level=0)
        else:
            returns_df.index = returns_df.index.astype(str)
            
    return model, processor, engineer, bias_map, jockey_stats, returns_df

def verify_hit(race_id, rec, returns_df):
    try:
        if isinstance(returns_df.index, pd.MultiIndex):
            if race_id not in returns_df.index.get_level_values(0): return 0
            race_rets = returns_df.loc[race_id]
        else:
            if race_id not in returns_df.index: return 0
            race_rets = returns_df.loc[[race_id]]
            
        payout = 0
        bet_type, method = rec.get('bet_type'), rec.get('method', 'SINGLE')
        # カラム0が馬券種
        hits = race_rets[race_rets[0] == bet_type]
        for _, h in hits.iterrows():
            try:
                money = int(str(h[2]).replace(',', '').replace('円', ''))
                wins = [int(x) for x in str(h[1]).replace('→', '-').split('-')]
                is_hit, bet_nums = False, set(rec.get('horse_numbers', []))
                if method == 'BOX': is_hit = set(wins).issubset(bet_nums)
                elif method in ['FORMATION', '流し']:
                    st = rec.get('formation', [])
                    if not st: is_hit = set(wins).issubset(bet_nums)
                    elif bet_type == '3連単':
                        if len(wins) == 3 and len(st) >= 3: is_hit = (wins[0] in st[0] and wins[1] in st[1] and wins[2] in st[2])
                    elif bet_type == '3連複':
                        if len(wins) == 3:
                            if len(st) == 2:
                                axis = set(st[0]).intersection(set(wins))
                                if len(axis) >= 1 and (set(wins) - axis).issubset(set(st[1])): is_hit = True
                            elif len(st) == 1: is_hit = set(wins).issubset(set(st[0]))
                    elif bet_type in ['馬連', 'ワイド']:
                        if len(wins) == 2:
                            if len(st) == 2:
                                w = set(wins)
                                axis = set(st[0]).intersection(w)
                                if len(axis) >= 1 and (w - axis).issubset(set(st[1])): is_hit = True
                            elif len(st) == 1: is_hit = set(wins).issubset(set(st[0]))
                elif method == 'SINGLE':
                    if bet_type == '単勝': is_hit = wins[0] in bet_nums
                    else: is_hit = list(wins) == list(rec.get('horse_numbers', []))
                if is_hit: payout += int(money * (rec.get('unit_amount', 100) / 100))
            except: continue
        return payout
    except: return 0

def get_dist_cat(dist):
    if dist == 0: return '不明'
    if dist <= 1400: return '短距離 (~1400)'
    if dist <= 1800: return 'マイル (1401~1800)'
    if dist <= 2200: return '中距離 (1801~2200)'
    return '長距離 (2201~)'

def to_md_table(df):
    cols = df.columns.tolist()
    header = "| " + " | ".join(['カテゴリ'] + cols) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    rows = []
    for idx, row in df.iterrows():
        r = "| " + str(idx) + " | " + " | ".join([f"{v:,}" if isinstance(v, (int, np.integer)) else f"{v:.1f}%" if 'rate' in str(df.columns[i]) or 'recovery' in str(df.columns[i]) else f"{v:.1f}" if isinstance(v, (float, np.float64)) else str(v) for i, v in enumerate(row)]) + " |"
        rows.append(r)
    return "\n".join([header, sep] + rows)

def run_analysis():
    print("=== 2025年 シミュレーション詳細分析（メタデータ完全復元版） ===")
    race_meta = create_race_info_map()
    model, processor, engineer, bias_map, jockey_stats, returns_df = load_resources()
    
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f: results = pickle.load(f)
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
    else:
        results['race_id'] = results.index.astype(str)
    
    results_2025 = results[results['race_id'].str.startswith('2025')].copy()
    unique_rids = results_2025['race_id'].unique().tolist()
    print(f"  - 抽出レース数: {len(unique_rids)}")
    
    results_2025.set_index('race_id', inplace=True)
    df_proc = processor.process_results(results_2025)
    df_proc.index = df_proc.index.astype(str)
    
    # 擬似日付（特徴量エンジニアリング用）
    rid_to_date = {rid: datetime(2025, 1, 1) + pd.to_timedelta(i, unit='D') for i, rid in enumerate(df_proc.index.unique())}
    df_proc['date'] = df_proc.index.map(rid_to_date)
    
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f: hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f: peds = pickle.load(f)
    
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc = engineer.add_jockey_features(df_proc, jockey_stats)[0]
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    df_proc = engineer.add_bias_features(df_proc, bias_map)
    df_proc = processor.encode_categorical(df_proc, ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam'])
    
    X = df_proc[model.feature_names].fillna(0).copy()
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
    df_proc['probability'] = model.predict(X.fillna(0))
    df_proc['expected_value'] = df_proc['probability'] * df_proc.get('単勝', 0)
    
    # Stats containers
    strat_names = ['formation', 'hybrid_1000']
    stats = {sn: [] for sn in strat_names}
    
    print("Simulating...", flush=True)
    # レース単位でループ
    # Note: feature engineering (merge) resets index, so use saved original_race_id
    group_col = 'original_race_id' if 'original_race_id' in df_proc.columns else 'race_id'
    
    # If using 'race_id' column, ensure it exists or restore from index if index is preserved
    if group_col not in df_proc.columns and df_proc.index.name != 'race_id':
        # fallback, try to find a column like race_id
        cols = [c for c in df_proc.columns if 'race_id' in str(c)]
        if cols: group_col = cols[0]
    
    # If still not found, assumes index is correct (which we know is False, but safety)
    if group_col in df_proc.columns:
        grouper = df_proc.groupby(group_col)
    else:
        grouper = df_proc.groupby(level=0)

    for race_id, race_df in tqdm(grouper):
        race_id = str(race_id)
        v_name = PLACE_DICT.get(race_id[4:6], '不明')
        info = race_meta.get(race_id, {'surface': '不明', 'distance': 0, 'entrants': 0, 'name': ''})
        
        # Infer Sex Condition from data
        # '性' column: 0=牡, 1=牝, 2=セ (defined in constants.py)
        # If all runners are Female (1), then it's Female Only.
        if '性' in race_df.columns:
            is_female_only = (race_df['性'] == 1).all()
        else:
            is_female_only = False
            
        sex_cat = '牝馬限定' if is_female_only else '混合/牡'

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
        df_preds = pd.DataFrame(preds)
        
        for sn in strat_names:
            budget = 5000 if sn == 'formation' else 1000
            recs = BettingAllocator.allocate_budget(df_preds, budget=budget, strategy=sn)
            if not recs: continue
            
            invest, payout = 0, 0
            for rec in recs:
                invest += rec.get('total_amount', 0)
                payout += verify_hit(race_id, rec, returns_df)
            
            stats[sn].append({
                'venue': v_name,
                'surface': info['surface'],
                'distance': get_dist_cat(info['distance']),
                'entrants': get_entrants_cat(info.get('entrants', 0)),
                'age': get_age_cat(info.get('name', '')),
                'class': get_class_cat(info.get('name', '')),
                'sex': sex_cat,
                'invest': invest,
                'payout': payout,
                'hit': 1 if payout > 0 else 0
            })
            
    # Report
    report = "# 2025年 シミュレーション詳細分析レポート\n\n- 対象: 2025年全レース\n- メタデータ: date_map_2025.pickle + horse_results.pickle より復元\n\n"
    for sn in strat_names:
        df_stats = pd.DataFrame(stats[sn])
        report += f"## 戦略: {sn}\n\n"
        if df_stats.empty:
            report += "対象データなし\n\n"
            continue
            
        for col, title in [
            ('venue', '開催場別'), ('surface', '馬場種別'), ('distance', '距離別'),
            ('entrants', '出走頭数別'), ('age', '馬齢別'), ('class', 'クラス別'), ('sex', '条件別')
        ]:
            # レース数と各指標の合計を計算
            agg_sum = df_stats.groupby(col).agg({'invest': 'sum', 'payout': 'sum', 'hit': 'sum'})
            agg_count = df_stats.groupby(col).size().to_frame('races')
            agg = pd.concat([agg_sum, agg_count], axis=1)
            
            agg['recovery'] = (agg['payout'] / agg['invest'] * 100).round(1)
            agg['hit_rate'] = (agg['hit'] / agg['races'] * 100).round(1)
            report += f"### {title}\n"
            report += to_md_table(agg[['races', 'invest', 'payout', 'recovery', 'hit_rate']]) + "\n\n"
            
    with open(REPORT_FILE, 'w', encoding='utf-8') as f: f.write(report)
    print(f"Done. Report: {REPORT_FILE}")

if __name__ == "__main__":
    run_analysis()
