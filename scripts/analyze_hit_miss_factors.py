"""
的中レース vs 非的中レース メタ分析
- 的中/非的中を分類し、各グループの特徴を比較
- 何が的中の要因となっているかを特定
"""
import os
import sys
import pickle
import random
import gc
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.training import HorseRaceModel, RacePredictor
from modules.betting_allocator import BettingAllocator

# === Config ===
TARGET_YEAR = 2025
STRATEGY = 'formation_flex'
BUDGET = 5000
MIN_CONFIDENCE = 'A'
SAMPLE_RATE = 0.1

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
CONFIDENCE_ORDER = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}


def load_resources():
    """共通リソースのロード"""
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, 'return_tables.pickle'), 'rb') as f:
        returns = pickle.load(f)
    
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    df_target = results[results['race_id'].astype(str).str.startswith('2025')].copy()
    active_horses = df_target['horse_id'].unique() if 'horse_id' in df_target.columns else []
    
    hr['date_str'] = hr['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
    hr['date'] = pd.to_datetime('20' + hr['date_str'], format='%Y/%m/%d', errors='coerce').dt.normalize()
    if len(active_horses) > 0:
        hr = hr[hr.index.isin(active_horses)].copy()
    
    returns['race_id_str'] = [str(x[0]) if isinstance(x, tuple) else str(x) for x in returns.index]
    returns_dict = {k: v for k, v in returns.groupby('race_id_str')}
    
    del results
    gc.collect()
    
    return df_target, hr, peds, returns_dict


def load_predictor():
    """モデル読み込み"""
    model = HorseRaceModel()
    model.load(os.path.join(HISTORICAL_MODEL_DIR, 'model.pkl'))
    
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(HISTORICAL_MODEL_DIR, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    
    return RacePredictor(model, processor, engineer)


def process_race(race_df, predictor, hr, peds):
    """特徴量構築"""
    try:
        df = race_df.copy()
        df.columns = df.columns.str.replace(' ', '')
        for col in ['馬番', '枠番', '単勝']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        df_proc = predictor.processor.process_results(df)
        df_proc = predictor.engineer.add_horse_history_features(df_proc, hr)
        df_proc = predictor.engineer.add_course_suitability_features(df_proc, hr)
        df_proc, _ = predictor.engineer.add_jockey_features(df_proc)
        df_proc = predictor.engineer.add_pedigree_features(df_proc, peds)
        df_proc = predictor.engineer.add_odds_features(df_proc)

        feature_names = predictor.model.feature_names
        X = pd.DataFrame(index=df_proc.index)
        for c in feature_names:
            if c in df_proc.columns:
                X[c] = pd.to_numeric(df_proc[c], errors='coerce').fillna(0)
            else:
                X[c] = 0

        probs = predictor.model.predict(X)
        
        df_res = df_proc.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = df_res['馬番']
        df_res['odds'] = pd.to_numeric(df_res.get('単勝', 10), errors='coerce').fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        return df_res
    except:
        return None


def calculate_confidence(race_df):
    """自信度を算出"""
    if race_df.empty: return 'D', {}
    
    sorted_df = race_df.sort_values('probability', ascending=False)
    top = sorted_df.iloc[0]
    
    top_prob = top['probability']
    top_ev = top['expected_value']
    top_odds = top['odds']
    
    # Top3の情報
    top3 = sorted_df.head(3)
    top3_probs = top3['probability'].tolist()
    top3_odds = top3['odds'].tolist()
    
    # 確率差 (1位-2位)
    prob_gap = top3_probs[0] - top3_probs[1] if len(top3_probs) > 1 else 0
    
    # 馬数
    num_horses = len(race_df)
    
    if top_prob >= 0.5 or top_ev >= 1.5: conf = 'S'
    elif top_prob >= 0.4 or top_ev >= 1.2: conf = 'A'
    elif top_prob >= 0.3 or top_ev >= 1.0: conf = 'B'
    elif top_prob >= 0.2: conf = 'C'
    else: conf = 'D'
    
    stats = {
        'confidence': conf,
        'top1_prob': top_prob,
        'top1_ev': top_ev,
        'top1_odds': top_odds,
        'top2_prob': top3_probs[1] if len(top3_probs) > 1 else 0,
        'top3_prob': top3_probs[2] if len(top3_probs) > 2 else 0,
        'top2_odds': top3_odds[1] if len(top3_odds) > 1 else 0,
        'top3_odds': top3_odds[2] if len(top3_odds) > 2 else 0,
        'prob_gap_1_2': prob_gap,
        'num_horses': num_horses,
        'prob_sum_top3': sum(top3_probs[:3]) if len(top3_probs) >= 3 else sum(top3_probs),
        'avg_odds': race_df['odds'].mean(),
        'max_odds': race_df['odds'].max(),
        'min_odds': race_df['odds'].min(),
    }
    
    return conf, stats


def verify_hit(race_id, rec, returns_dict):
    """的中検証と払戻額を返す"""
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None:
        return 0, None
    
    payout = 0
    bet_type = rec.get('type') or rec.get('bet_type')
    bet_horse_nums = set(rec.get('horses', []) or rec.get('horse_numbers', []))
    
    if not bet_type or not bet_horse_nums:
        return 0, None
    
    hit_details = None
    
    try:
        hits = race_rets[race_rets[0] == bet_type]

        for _, h in hits.iterrows():
            try:
                money = int(str(h[2]).replace(',','').replace('円',''))
                win_str = str(h[1]).replace('→','-')
                if '-' in win_str: win_nums = [int(x) for x in win_str.split('-')]
                else: win_nums = [int(win_str)]
                
                is_hit = False
                if bet_type in ['単勝', '複勝']:
                    if win_nums[0] in bet_horse_nums: is_hit = True
                elif bet_type in ['馬連', 'ワイド', '3連複', '3連単']:
                    method = rec.get('method')
                    formation = rec.get('formation')
                    
                    if method in ['流し', 'NAGASHI', '1軸流し', 'FORMATION'] and formation and len(formation) >= 2:
                        axis = set(formation[0])
                        opponents = set(formation[1])
                        win_set = set(win_nums)
                        if axis.issubset(win_set):
                            remaining = win_set - axis
                            if remaining.issubset(opponents):
                                is_hit = True
                    elif method == 'BOX':
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                    else:
                        if set(win_nums).issubset(bet_horse_nums): is_hit = True
                    
                if is_hit:
                    unit = rec.get('unit_amount', 100)
                    if unit == 0:
                        cnt = rec.get('count', rec.get('points', 0))
                        if cnt > 0: unit = rec.get('amount', rec.get('total_amount', 0)) // cnt
                        else: unit = 100
                    
                    pay = money * (unit / 100)
                    payout += pay
                    hit_details = {
                        'bet_type': bet_type,
                        'payout': pay,
                        'odds_actual': money / 100
                    }

            except:
                pass
    except:
        pass
    
    return payout, hit_details


def extract_race_metadata(race_id, race_df):
    """レースのメタデータを抽出"""
    metadata = {
        'race_id': str(race_id),
        'month': int(str(race_id)[4:6]) if len(str(race_id)) >= 6 else 0,
        'venue': int(str(race_id)[4:6]) if len(str(race_id)) >= 6 else 0,
    }
    
    # レースの基本情報があれば追加
    if 'コース' in race_df.columns:
        course = race_df['コース'].iloc[0] if len(race_df) > 0 else ''
        metadata['is_turf'] = 1 if '芝' in str(course) else 0
        metadata['is_dirt'] = 1 if 'ダ' in str(course) else 0
    
    return metadata


def run_analysis():
    print("="*60)
    print("的中/非的中 メタ分析")
    print(f"戦略: {STRATEGY} / 予算: ¥{BUDGET} / 閾値: {MIN_CONFIDENCE}以上")
    print("="*60)
    
    # データ読み込み
    print("\nLoading data...")
    df_target, hr, peds, returns_dict = load_resources()
    
    # サンプル抽出
    race_ids = df_target['race_id'].unique().tolist()
    sample_size = max(1, int(len(race_ids) * SAMPLE_RATE))
    sample_ids = random.sample(race_ids, sample_size)
    print(f"Sample: {sample_size}/{len(race_ids)} races")
    
    # モデル読み込み
    print("Loading model...")
    predictor = load_predictor()
    
    # 分析データ収集
    hit_data = []  # 的中レース
    miss_data = []  # 非的中レース
    
    print("\nAnalyzing races...")
    for i, race_id in enumerate(sample_ids):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(sample_ids)}")
            
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty:
                continue
            
            df_pred = process_race(race_df, predictor, hr, peds)
            if df_pred is None or df_pred.empty:
                continue
            
            conf, stats = calculate_confidence(df_pred)
            
            # 閾値チェック
            if CONFIDENCE_ORDER.get(conf, 99) > CONFIDENCE_ORDER.get(MIN_CONFIDENCE, 1):
                continue
            
            # メタデータ
            metadata = extract_race_metadata(race_id, race_df)
            
            # ベット生成
            recs = BettingAllocator.allocate_budget(df_pred, BUDGET, strategy=STRATEGY)
            if not recs:
                continue
            
            # レースごとの的中判定
            race_hit = False
            race_payout = 0
            
            for rec in recs:
                payout, hit_details = verify_hit(race_id, rec, returns_dict)
                if payout > 0:
                    race_hit = True
                    race_payout += payout
            
            # データ記録
            record = {**stats, **metadata, 'total_payout': race_payout}
            
            if race_hit:
                hit_data.append(record)
            else:
                miss_data.append(record)
                
        except Exception as e:
            continue
    
    # 分析結果
    print("\n" + "="*60)
    print("分析結果")
    print("="*60)
    
    df_hit = pd.DataFrame(hit_data)
    df_miss = pd.DataFrame(miss_data)
    
    print(f"\n的中レース: {len(df_hit)}")
    print(f"非的中レース: {len(df_miss)}")
    print(f"的中率: {len(df_hit) / (len(df_hit) + len(df_miss)) * 100:.1f}%")
    
    # 統計比較
    compare_cols = ['top1_prob', 'top1_ev', 'top1_odds', 'prob_gap_1_2', 
                    'num_horses', 'prob_sum_top3', 'avg_odds', 'max_odds']
    
    print("\n" + "-"*60)
    print("特徴量比較 (平均値)")
    print("-"*60)
    print(f"{'特徴量':<20} {'的中':>12} {'非的中':>12} {'差':>12}")
    print("-"*60)
    
    insights = []
    for col in compare_cols:
        if col in df_hit.columns and col in df_miss.columns:
            hit_mean = df_hit[col].mean()
            miss_mean = df_miss[col].mean()
            diff = hit_mean - miss_mean
            diff_pct = (diff / miss_mean * 100) if miss_mean != 0 else 0
            
            print(f"{col:<20} {hit_mean:>12.3f} {miss_mean:>12.3f} {diff:>+12.3f}")
            
            if abs(diff_pct) > 10:
                insights.append({
                    'feature': col,
                    'hit_mean': hit_mean,
                    'miss_mean': miss_mean,
                    'diff_pct': diff_pct
                })
    
    # 自信度分布
    print("\n" + "-"*60)
    print("自信度分布")
    print("-"*60)
    print(f"{'自信度':<10} {'的中':>10} {'非的中':>10} {'的中率':>10}")
    for conf in ['S', 'A', 'B']:
        hit_cnt = len(df_hit[df_hit['confidence'] == conf]) if 'confidence' in df_hit.columns else 0
        miss_cnt = len(df_miss[df_miss['confidence'] == conf]) if 'confidence' in df_miss.columns else 0
        total = hit_cnt + miss_cnt
        rate = hit_cnt / total * 100 if total > 0 else 0
        print(f"{conf:<10} {hit_cnt:>10} {miss_cnt:>10} {rate:>9.1f}%")
    
    # 払戻額分析
    print("\n" + "-"*60)
    print("的中時の払戻額分析")
    print("-"*60)
    if 'total_payout' in df_hit.columns and len(df_hit) > 0:
        payouts = df_hit['total_payout']
        print(f"平均: ¥{payouts.mean():,.0f}")
        print(f"中央値: ¥{payouts.median():,.0f}")
        print(f"最大: ¥{payouts.max():,.0f}")
        print(f"最小: ¥{payouts.min():,.0f}")
        
        # 高額的中の特徴
        high_payout = df_hit[df_hit['total_payout'] >= payouts.quantile(0.9)]
        if len(high_payout) > 0:
            print(f"\n上位10%的中 ({len(high_payout)}件) の特徴:")
            for col in ['top1_prob', 'top1_ev', 'top1_odds', 'prob_gap_1_2']:
                if col in high_payout.columns:
                    print(f"  {col}: {high_payout[col].mean():.3f} (全体: {df_hit[col].mean():.3f})")
    
    # キーインサイト
    print("\n" + "="*60)
    print("キーインサイト")
    print("="*60)
    
    for insight in insights:
        direction = "高い" if insight['diff_pct'] > 0 else "低い"
        print(f"・{insight['feature']}: 的中レースは非的中より{abs(insight['diff_pct']):.1f}%{direction}")
    
    # CSV出力
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    df_hit.to_csv(os.path.join(output_dir, 'hit_races.csv'), index=False, encoding='utf-8-sig')
    df_miss.to_csv(os.path.join(output_dir, 'miss_races.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n詳細データ保存: {output_dir}/")
    print("  - hit_races.csv")
    print("  - miss_races.csv")


if __name__ == "__main__":
    run_analysis()
