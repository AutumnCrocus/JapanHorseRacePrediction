"""
CatBoost 多頭数BOX (BOX4, BOX5, BOX6) 2025年シミュレーション
- 期待値(EV >= 2.5)フィルター適用
- 不的中時の乖離分析（抜け頭数・抜け馬の予測順位）を集計
"""

import os
import sys
import pickle
import json
import itertools
from datetime import datetime
from collections import Counter

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.constants import RAW_DATA_DIR, MODEL_DIR, PROCESSED_DATA_DIR
from modules.betting_allocator import BettingAllocator

# ===== 設定 =====
BUDGET = 5000
EV_THRESHOLD = 2.5
STRATEGIES = ['box4_sanrenpuku', 'box5_sanrenpuku', 'box6_sanrenpuku']
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

class CatBoostWrapper:
    def __init__(self, data_dict: dict):
        self._model = data_dict['model']
        self.feature_names = data_dict['feature_names']
        
    def predict(self, X):
        import catboost as cb
        X_aligned = X[self.feature_names].fillna(0)
        return self._model.predict_proba(X_aligned)[:, 1]

def load_payouts_2025() -> dict:
    """2025年の配当データをロード"""
    path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
    with open(path, 'rb') as f:
        rt = pickle.load(f)

    payouts = {}
    ride_ids_all = rt.index.get_level_values(0).astype(str)
    target_ids = rt.index[ride_ids_all.str.startswith('2025')].get_level_values(0).unique()

    for rid in target_ids:
        group = rt.loc[rid]
        if '3連複' not in group.iloc[:, 0].values:
            continue
            
        # 3連複の配当だけ抽出
        sanrenpuku_row = group[group.iloc[:, 0] == '3連複'].iloc[0]
        try:
            combo_str = str(sanrenpuku_row.iloc[1]).replace('→', '-').replace(' ', '-')
            combo = tuple(sorted(int(x) for x in combo_str.split('-')))
            pay_amt = int(str(sanrenpuku_row.iloc[2]).replace(',', '').replace('円', ''))
            payouts[str(rid)] = {'sanrenpuku': {combo: pay_amt}}
        except Exception:
            pass

    return payouts

def load_data():
    """前処理済みデータセットと配当データをロード"""
    print("[1] 前処理済みデータセットを読み込み中...", flush=True)
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    with open(dataset_path, 'rb') as f:
        ds = pickle.load(f)

    df = ds['data']
    feature_names = ds.get('feature_names', [])

    if 'original_race_id' in df.columns:
        df['race_id'] = df['original_race_id'].astype(str)
    else:
        df['race_id'] = df.index.astype(str)

    df_2025 = df[df['year'] == 2025].copy() if 'year' in df.columns else df.copy()
    print(f"   2025年データ: {len(df_2025)}行, レース数: {len(df_2025['race_id'].unique())}", flush=True)

    print("[2]配当データを読み込み中...", flush=True)
    payouts = load_payouts_2025()
    print(f"   配当データ件数: {len(payouts)}レース", flush=True)

    return df_2025, payouts

def load_model():
    print("[3] CatBoost モデルを読み込み中...", flush=True)
    cat_path = os.path.join(MODEL_DIR, 'catboost_2010_2024', 'model.pkl')
    with open(cat_path, 'rb') as f:
        cat_data = pickle.load(f)
    print("   ロード完了", flush=True)
    return CatBoostWrapper(cat_data)

def predict_race(race_df: pd.DataFrame, model) -> pd.DataFrame | None:
    try:
        X = pd.DataFrame(index=race_df.index)
        for c in model.feature_names:
            X[c] = pd.to_numeric(race_df[c], errors='coerce').fillna(0) if c in race_df.columns else 0.0

        probs = model.predict(X)

        df_res = race_df.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = pd.to_numeric(
            df_res.get('馬番', pd.Series(range(1, len(df_res) + 1))),
            errors='coerce'
        ).fillna(0).astype(int)
        df_res['odds'] = pd.to_numeric(
            df_res.get('単勝', pd.Series([10.0] * len(df_res))),
            errors='coerce'
        ).fillna(10.0)

        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        
        # 予測確率で降順ソートしておく
        df_res = df_res.sort_values('probability', ascending=False).reset_index(drop=True)
        # 予測順位(evaluation rank)を付与 (1位〜)
        df_res['eval_rank'] = df_res.index + 1

        return df_res
    except Exception:
        return None

def compute_race_ev(df_preds: pd.DataFrame) -> float:
    top4_ev = df_preds['expected_value'].head(4)
    if top4_ev.empty: return 0.0
    return float(top4_ev.mean())

def get_winning_combo(race_pay: dict) -> tuple | None:
    sanrenpuku_pay = race_pay.get('sanrenpuku', {})
    if not sanrenpuku_pay: return None
    return list(sanrenpuku_pay.keys())[0]

def verify_hit(recs: list, winning_combo: tuple, race_pay: dict) -> int:
    """的中・配当計算"""
    if not winning_combo or not recs: return 0
    
    total_payout = 0
    sanrenpuku_pay = race_pay.get('sanrenpuku', {})
    
    for rec in recs:
        unit_amt = rec.get('unit_amount', 100) or 100
        comb = rec.get('combination', '')
        if comb and isinstance(comb, str) and 'BOX' in comb:
            comb = comb.split(' BOX')[0]
            bet_horses = [int(x) for x in comb.split(',') if x.isdigit()]
        else:
            bet_horses = [int(h) for h in rec.get('horse_numbers', rec.get('horses', [])) if str(h).isdigit()]
        
        # BOX組み合わせ生成
        bought = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 3)]
        
        for combo in bought:
            if combo == winning_combo:
                pay = sanrenpuku_pay.get(combo, 0)
                total_payout += int(pay * (unit_amt / 100))
                
    return total_payout

def analyze_miss(recs: list, winning_combo: tuple, df_preds: pd.DataFrame) -> dict:
    """"不的中時の乖離幅を分析"""
    if not winning_combo or not recs: return None
    
    # 馬番から評価順位を引くための辞書
    rank_map = {row['horse_number']: row['eval_rank'] for _, row in df_preds.iterrows()}
    
    # 購入した馬番の集合
    purchased_horses = set()
    for rec in recs:
        # format_recommendationsを経由した場合の考慮
        comb = rec.get('combination', '')
        if comb and isinstance(comb, str) and 'BOX' in comb:
            comb = comb.split(' BOX')[0]
            bet_horses = [int(x) for x in comb.split(',') if x.isdigit()]
        else:
            bet_horses = [int(h) for h in rec.get('horse_numbers', rec.get('horses', [])) if str(h).isdigit()]
            
        purchased_horses.update(bet_horses)
        
    # 実際の3着内馬(正解)
    winning_set = set(winning_combo)
    
    # 買ってないのに3着内に入った馬
    missed_horses = winning_set - purchased_horses
    
    missed_count = len(missed_horses)
    missed_ranks = [rank_map.get(m, 99) for m in missed_horses]
    
    return {
        'missed_count': missed_count,      # 抜け頭数 (0=的中または抜けなし, 1, 2, 3)
        'missed_ranks': missed_ranks       # 抜け馬の評価順位 (例: [6, 8])
    }

def build_summary_markdown(stats, df_results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORT_DIR, f'simulation_catboost_multibox_2025_{timestamp}.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# CatBoost 多頭数BOXの2025年シミュレーション結果\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- モデル: `CatBoost`\n")
        f.write(f"- フィルタ: **EV >= {EV_THRESHOLD}**\n")
        f.write(f"- 予算: 各レース最大 {BUDGET}円\n\n")
        
        f.write("---\n\n## 1. 総合パフォーマンス比較\n\n")
        f.write("| 戦略 | 回収率 | トータル収益(円) | 投資総額(円) | 的中率 | 投票R数 | 的中R数 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for idx, row in df_results.iterrows():
            strat = row['strategy']
            f.write(
                f"| {strat} "
                f"| **{row['recovery']:.1f}%** "
                f"| {int(row['profit']):+,} "
                f"| {int(row['bet']):,} "
                f"| {row['hit_rate']:.1f}% "
                f"| {int(row['races'])} "
                f"| {int(row['hits'])} |\n"
            )
            
        f.write("\n---\n\n## 2. 不的中レースの乖離分析（抜け目の傾向）\n\n")
        f.write("不的中となったレースについて、実際の3着内馬(正解)の組み合わせから**何頭買えていなかったか（抜け頭数）**、およびその**抜け馬の予測順位**を集計しました。\n\n")
        
        for strat in STRATEGIES:
            s = stats[strat]
            miss_total = s['races'] - s['hits']
            if miss_total == 0: continue
            
            f.write(f"### {strat} の抜け目傾向\n")
            counts = s['miss_counts']
            f.write(f"- **不的中レース数**: {miss_total}R\n")
            f.write(f"  - 1頭抜け: {counts.get(1, 0)}回 ({counts.get(1, 0)/miss_total:.1%})\n")
            f.write(f"  - 2頭抜け: {counts.get(2, 0)}回 ({counts.get(2, 0)/miss_total:.1%})\n")
            f.write(f"  - 3頭全抜け: {counts.get(3, 0)}回 ({counts.get(3, 0)/miss_total:.1%})\n")
            
            ranks = s['miss_ranks']
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                f.write(f"- **抜け馬の平均予測順位**: {avg_rank:.1f}位\n")
                f.write("- **抜け馬の順位分布**:\n")
                
                # 順位分布を出力
                rank_counts = Counter(ranks)
                for r in sorted(rank_counts.keys()):
                    if r <= 15: # 15位以下は表示
                        f.write(f"  - {r}位: {rank_counts[r]}頭\n")
            f.write("\n")
            
    return report_path


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"=== CatBoost 多頭数BOX 2025年シミュレーション開始: {timestamp} ===\n", flush=True)

    df_2025, payouts = load_data()
    model = load_model()

    race_ids = sorted(df_2025['race_id'].unique().tolist())
    
    # 統計用コンテナ
    stats = {}
    for st in STRATEGIES:
        stats[st] = {
            'bet': 0, 'ret': 0, 'hits': 0, 'races': 0,
            'miss_counts': Counter(), 'miss_ranks': []
        }

    for race_id in tqdm(race_ids, desc="Simulating"):
        race_df = df_2025[df_2025['race_id'] == race_id]
        if len(race_df) < 5: continue

        race_pay = payouts.get(str(race_id))
        winning_combo = get_winning_combo(race_pay) if race_pay else None

        df_preds = predict_race(race_df, model)
        if df_preds is None or df_preds.empty: continue

        # EVフィルター適用
        race_ev = compute_race_ev(df_preds)
        if race_ev < EV_THRESHOLD: continue

        # 戦略ごとの処理
        for strat in STRATEGIES:
            try:
                recs = BettingAllocator.allocate_budget(df_preds, BUDGET, strategy=strat)
            except Exception:
                continue
                
            if not recs: continue
            
            # 各戦略の推奨買い目を足し合わせるか（今回は単一のBOXなので合計金額）
            total_amt = sum(r.get('total_amount', r.get('amount', 0)) for r in recs)
            if total_amt == 0: continue
            
            stats[strat]['races'] += 1
            stats[strat]['bet'] += total_amt
            
            pay = verify_hit(recs, winning_combo, race_pay) if race_pay else 0
            stats[strat]['ret'] += pay
            
            if pay > 0:
                stats[strat]['hits'] += 1
            else:
                # 不的中レースの分析
                miss_info = analyze_miss(recs, winning_combo, df_preds)
                if miss_info and miss_info['missed_count'] > 0:
                    stats[strat]['miss_counts'][miss_info['missed_count']] += 1
                    stats[strat]['miss_ranks'].extend(miss_info['missed_ranks'])

    # ===== 結果の集計とレポート作成 =====
    results_list = []
    for strat, v in stats.items():
        recov = (v['ret'] / v['bet'] * 100) if v['bet'] > 0 else 0.0
        profit = v['ret'] - v['bet']
        hit_rate = (v['hits'] / v['races'] * 100) if v['races'] > 0 else 0.0
        results_list.append({
            'strategy': strat,        'recovery': recov,
            'profit': profit,         'bet': v['bet'],
            'ret': v['ret'],          'hit_rate': hit_rate,
            'races': v['races'],      'hits': v['hits']
        })

    df_results = pd.DataFrame(results_list)
    
    report_path = build_summary_markdown(stats, df_results)

    print(f"\n=== シミュレーション完了 ===")
    print(f"レポート: {report_path}")
    print(df_results[['strategy', 'recovery', 'profit', 'hit_rate', 'races']].to_string(index=False))


if __name__ == '__main__':
    main()
