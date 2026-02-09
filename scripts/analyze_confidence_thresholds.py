"""
自信度閾値別シミュレーション分析
- ヒストリカルモデルのみで S, A, B の各閾値をテスト
- 結果を比較して最適な閾値を特定
"""
import os
import sys
import json
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
SAMPLE_RATE = 0.1  # 10%サンプル

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLDS = ['S', 'A', 'B']
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
    if race_df.empty: return 'D'
    top = race_df.sort_values('probability', ascending=False).iloc[0]
    
    top_prob = top['probability']
    top_ev = top['expected_value']
    
    if top_prob >= 0.5 or top_ev >= 1.5: return 'S'
    elif top_prob >= 0.4 or top_ev >= 1.2: return 'A'
    elif top_prob >= 0.3 or top_ev >= 1.0: return 'B'
    elif top_prob >= 0.2: return 'C'
    else: return 'D'


def verify_hit(race_id, rec, returns_dict):
    """的中検証"""
    race_rets = returns_dict.get(str(race_id))
    if race_rets is None:
        return 0
    
    payout = 0
    bet_type = rec.get('type') or rec.get('bet_type')
    bet_horse_nums = set(rec.get('horses', []) or rec.get('horse_numbers', []))
    
    if not bet_type or not bet_horse_nums:
        return 0
    
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

            except:
                pass
    except:
        pass
    
    return payout


def simulate_with_confidence(predictor, hr, peds, df_target, returns_dict, 
                              race_ids, min_confidence: str):
    """指定した自信度閾値でシミュレーション"""
    total_bet = 0
    total_return = 0
    race_count = 0
    bet_count = 0
    hit_count = 0
    
    for race_id in race_ids:
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty:
                continue
            
            df_pred = process_race(race_df, predictor, hr, peds)
            if df_pred is None or df_pred.empty:
                continue
            
            conf = calculate_confidence(df_pred)
            
            # 閾値チェック
            if CONFIDENCE_ORDER.get(conf, 99) > CONFIDENCE_ORDER.get(min_confidence, 1):
                continue
            
            race_count += 1
            
            recs = BettingAllocator.allocate_budget(df_pred, BUDGET, strategy=STRATEGY)
            if not recs:
                continue
            
            for rec in recs:
                bet_amount = rec.get('total_amount', rec.get('amount', 100))
                total_bet += bet_amount
                bet_count += 1
                
                ret = verify_hit(race_id, rec, returns_dict)
                total_return += ret
                if ret > 0:
                    hit_count += 1
                    
        except:
            continue
    
    recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0
    
    return {
        'min_confidence': min_confidence,
        'race_count': race_count,
        'bet_count': bet_count,
        'hit_count': hit_count,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'recovery_rate': recovery_rate,
        'profit': total_return - total_bet,
        'avg_payout': total_return / hit_count if hit_count > 0 else 0
    }


def main():
    print("="*60)
    print("自信度閾値別シミュレーション分析")
    print(f"戦略: {STRATEGY} / 予算: ¥{BUDGET} / サンプル: {int(SAMPLE_RATE*100)}%")
    print("="*60)
    
    # データ読み込み
    print("\nLoading data...")
    df_target, hr, peds, returns_dict = load_resources()
    total_races = df_target['race_id'].nunique()
    
    # サンプル抽出
    race_ids = df_target['race_id'].unique().tolist()
    sample_size = max(1, int(len(race_ids) * SAMPLE_RATE))
    sample_ids = random.sample(race_ids, sample_size)
    print(f"Sample: {sample_size}/{total_races} races")
    
    # モデル読み込み
    print("Loading model...")
    predictor = load_predictor()
    
    # 各閾値でシミュレーション
    results = []
    for conf_threshold in CONFIDENCE_THRESHOLDS:
        print(f"\n--- 自信度 {conf_threshold}以上 ---")
        result = simulate_with_confidence(
            predictor, hr, peds, df_target, returns_dict, 
            sample_ids, conf_threshold
        )
        results.append(result)
        print(f"レース数: {result['race_count']}, ベット回数: {result['bet_count']}")
        print(f"的中率: {result['hit_rate']:.1f}%, 回収率: {result['recovery_rate']:.1f}%")
        print(f"平均的中単価: ¥{result['avg_payout']:,.0f}")
    
    # 結果サマリー
    print("\n" + "="*60)
    print("結果サマリー")
    print("="*60)
    print(f"{'閾値':<8} {'レース数':>10} {'回収率':>10} {'的中率':>10} {'平均的中単価':>15}")
    print("-"*60)
    for r in results:
        print(f"{r['min_confidence']+'以上':<8} {r['race_count']:>10} {r['recovery_rate']:>9.1f}% {r['hit_rate']:>9.1f}% ¥{r['avg_payout']:>14,.0f}")
    print("-"*60)
    
    # 最適閾値
    best = max(results, key=lambda x: x['recovery_rate'])
    print(f"\n✓ 最適閾値: {best['min_confidence']}以上 (回収率 {best['recovery_rate']:.1f}%)")
    
    # ファイル出力
    output_path = os.path.join(os.path.dirname(__file__), '..', 'confidence_threshold_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'strategy': STRATEGY,
            'budget': BUDGET,
            'sample_rate': SAMPLE_RATE,
            'sample_size': sample_size,
            'total_races': total_races,
            'results': results,
            'best_threshold': best['min_confidence']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果保存: {output_path}")


if __name__ == "__main__":
    main()
