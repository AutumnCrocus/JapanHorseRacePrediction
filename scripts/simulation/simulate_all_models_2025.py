"""
2025年 全モデル×全戦略 高速シミュレーション
- 前処理済みデータ(dataset_2010_2025.pkl)を使用して高速化
- モデル: lgbm, ltr, stacking
- 戦略: box4_sanrenpuku, ranking_anchor, wide_nagashi, sanrenpuku_1axis, formation
- 予算: 5000円/レース
- 処理: 逐次（並列処理なし）
"""
import os
import sys
import pickle
import json
import itertools
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from modules.constants import RAW_DATA_DIR, MODEL_DIR, PROCESSED_DATA_DIR
from modules.training import HorseRaceModel
from modules.betting_allocator import BettingAllocator

# ===== 設定 =====
BUDGET = 5000
STRATEGIES = ['box4_sanrenpuku', 'ranking_anchor', 'wide_nagashi', 'sanrenpuku_1axis', 'formation']
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

BET_KEY_MAP = {
    '単勝': 'tan', '複勝': 'fuku', '馬連': 'umaren', 'ワイド': 'wide',
    '馬単': 'umatan', '3連複': 'sanrenpuku', '3連単': 'sanrentan'
}


# ===== モデルラッパー =====
class LTRWrapper:
    def __init__(self, data_dict: dict):
        self._model = data_dict['model']
        self.feature_names = data_dict['feature_names']
        self.model_type = 'ltr'

    def predict(self, X):
        return self._model.predict(X[self.feature_names])


# ===== 配当データのロード =====
def load_payouts_2025() -> dict:
    path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
    with open(path, 'rb') as f:
        rt = pickle.load(f)

    payouts = {}
    ride_ids_all = rt.index.get_level_values(0).astype(str)
    target_ids = rt.index[ride_ids_all.str.startswith('2025')].get_level_values(0).unique()

    for rid in target_ids:
        group = rt.loc[rid]
        race_pay = {}
        for _, row in group.iterrows():
            btype = str(row.iloc[0])
            bk = BET_KEY_MAP.get(btype)
            if not bk:
                continue
            try:
                pay_amt = int(str(row.iloc[2]).replace(',', '').replace('円', ''))
                combo_str = str(row.iloc[1]).replace('→', '-')
                if '-' in combo_str:
                    combo = tuple(sorted(int(x) for x in combo_str.split('-')))
                else:
                    combo = int(combo_str)
                if bk not in race_pay:
                    race_pay[bk] = {}
                race_pay[bk][combo] = pay_amt
            except Exception:
                pass
        if race_pay:
            payouts[str(rid)] = race_pay

    return payouts


# ===== データ読み込み =====
def load_data():
    print("[1] 前処理済みデータセットを読み込み中...", flush=True)
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    with open(dataset_path, 'rb') as f:
        ds = pickle.load(f)

    df = ds['data']
    lgbm_features = ds.get('feature_names', [])

    # race_id列を確定
    if 'original_race_id' in df.columns:
        df['race_id'] = df['original_race_id'].astype(str)
    elif 'race_id' not in df.columns:
        df['race_id'] = df.index.astype(str)

    # 2025年のみ
    df_2025 = df[df['year'] == 2025].copy() if 'year' in df.columns else df.copy()
    print(f"   2025年データ: {len(df_2025)}行, レース数: {len(df_2025['race_id'].unique())}", flush=True)

    print("[2] 配当データを読み込み中...", flush=True)
    payouts = load_payouts_2025()
    print(f"   配当データ件数: {len(payouts)}レース", flush=True)

    return df_2025, lgbm_features, payouts


# ===== モデル読み込み =====
def load_models():
    print("[3] モデルを読み込み中...", flush=True)
    models = {}

    # --- LGBM (historical_2010_2024) ---
    try:
        lgbm_dir = os.path.join(MODEL_DIR, 'historical_2010_2024')
        m = HorseRaceModel()
        m.load(os.path.join(lgbm_dir, 'model.pkl'))
        models['lgbm'] = ('lgbm', m)
        print("   [lgbm] ロード完了", flush=True)
    except Exception as e:
        print(f"   [lgbm] ロード失敗: {e}", flush=True)

    # --- LTR (standalone_ranking) ---
    try:
        ltr_path = os.path.join(MODEL_DIR, 'standalone_ranking', 'ranking_model.pkl')
        with open(ltr_path, 'rb') as f:
            ltr_data = pickle.load(f)
        models['ltr'] = ('ltr', LTRWrapper(ltr_data))
        print("   [ltr] ロード完了", flush=True)
    except Exception as e:
        print(f"   [ltr] ロード失敗: {e}", flush=True)

    # --- Stacking (experiment_model_2026) ---
    try:
        stacking_path = os.path.join(MODEL_DIR, 'experiment_model_2026.pkl')
        m_st = HorseRaceModel()
        m_st.load(stacking_path)
        models['stacking'] = ('stacking', m_st)
        print("   [stacking] ロード完了", flush=True)
    except Exception as e:
        print(f"   [stacking] ロード失敗: {e}", flush=True)

    return models


# ===== 予測実行 =====
def predict_race(race_df: pd.DataFrame, model_type: str, model, feature_names: list):
    """前処理済みデータからレース予測を行う"""
    try:
        X = pd.DataFrame(index=race_df.index)
        for c in feature_names:
            if c in race_df.columns:
                X[c] = pd.to_numeric(race_df[c], errors='coerce').fillna(0)
            else:
                X[c] = 0.0

        # deepfm_scoreが必要でなければスキップ
        if 'deepfm_score' in feature_names and 'deepfm_score' not in race_df.columns:
            X['deepfm_score'] = 0.5

        probs = model.predict(X)

        # LTRは正規化
        if model_type == 'ltr':
            mn, mx = probs.min(), probs.max()
            probs = (probs - mn) / (mx - mn) if mx != mn else np.full(len(probs), 0.5)

        df_res = race_df.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = pd.to_numeric(df_res.get('馬番', pd.Series(range(1, len(df_res)+1))), errors='coerce').fillna(0).astype(int)
        df_res['odds'] = pd.to_numeric(df_res.get('単勝', pd.Series([10.0]*len(df_res))), errors='coerce').fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']

        return df_res
    except Exception:
        return None


# ===== 的中判定 =====
def verify_hit(rec: dict, race_pay: dict) -> int:
    bet_key = BET_KEY_MAP.get(rec.get('bet_type', ''))
    if not bet_key or bet_key not in race_pay:
        return 0

    winning_data = race_pay[bet_key]
    method = rec.get('method', '')
    bet_horses = []
    for h in rec.get('horse_numbers', []):
        try:
            bet_horses.append(int(h))
        except Exception:
            pass

    unit_amt = rec.get('unit_amount', 100) or 100

    bought = []
    if method == 'BOX':
        if bet_key in ['umaren', 'wide']:
            bought = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 2)]
        elif bet_key == 'sanrenpuku':
            bought = [tuple(sorted(c)) for c in itertools.combinations(bet_horses, 3)]
    elif method in ['流し', 'FORMATION', 'Formation']:
        structure = rec.get('formation', [])
        if not structure:
            structure = [bet_horses]
        if bet_key in ['umaren', 'wide'] and len(structure) >= 2:
            for h1 in structure[0]:
                for h2 in structure[1]:
                    if int(h1) != int(h2):
                        bought.append(tuple(sorted((int(h1), int(h2)))))
        elif bet_key == 'sanrenpuku' and structure:
            axis = [int(h) for h in structure[0]]
            opps = [int(h) for h in structure[1]] if len(structure) > 1 else []
            for pair in itertools.combinations(opps, 2):
                for ax in axis:
                    bought.append(tuple(sorted([ax, pair[0], pair[1]])))
    else:
        if bet_key in ['tan', 'fuku']:
            for h in bet_horses:
                val = winning_data.get(h, 0)
                if val > 0:
                    return int(val * (unit_amt / 100))
            return 0
        elif bet_key in ['umaren', 'wide'] and len(bet_horses) >= 2:
            bought = [tuple(sorted(bet_horses[:2]))]
        elif bet_key == 'sanrenpuku' and len(bet_horses) >= 3:
            bought = [tuple(sorted(bet_horses[:3]))]

    payout = 0
    for combo in set(bought):
        val = winning_data.get(combo, 0)
        if val > 0:
            payout += int(val * (unit_amt / 100))
    return payout


# ===== メイン =====
def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"=== 2025年 全モデル×戦略 高速シミュレーション開始: {timestamp} ===\n", flush=True)

    df_2025, lgbm_features, payouts = load_data()
    models = load_models()

    if not models:
        print("ERROR: モデルを1つも読み込めませんでした。")
        return

    race_ids = sorted(df_2025['race_id'].unique().tolist())
    print(f"\n対象レース: {len(race_ids)}件, 予算: {BUDGET}円, モデル: {list(models.keys())}\n", flush=True)

    combo_keys = [(name, strat) for name in models.keys() for strat in STRATEGIES]
    stats = {k: {'bet': 0, 'ret': 0, 'hits': 0, 'races': 0} for k in combo_keys}

    for race_id in tqdm(race_ids, desc="Simulating", mininterval=3.0):
        race_df = df_2025[df_2025['race_id'] == race_id]
        if len(race_df) < 4:
            continue

        race_pay = payouts.get(str(race_id))

        for m_name, (model_type, model) in models.items():
            # モデルの特徴量リストを取得
            if hasattr(model, 'feature_names'):
                feat = model.feature_names
            else:
                feat = lgbm_features

            df_preds = predict_race(race_df, model_type, model, feat)
            if df_preds is None or df_preds.empty:
                continue

            df_sorted = df_preds.sort_values('probability', ascending=False)

            for strat in STRATEGIES:
                key = (m_name, strat)
                try:
                    recs = BettingAllocator.allocate_budget(df_sorted, BUDGET, strategy=strat)
                except Exception:
                    continue

                if not recs:
                    continue

                stats[key]['races'] += 1
                race_hit = False
                for rec in recs:
                    amt = rec.get('total_amount', 0)
                    stats[key]['bet'] += amt
                    if race_pay:
                        pay = verify_hit(rec, race_pay)
                        stats[key]['ret'] += pay
                        if pay > 0:
                            race_hit = True
                if race_hit:
                    stats[key]['hits'] += 1

    # ===== レポート生成 =====
    results_list = []
    for (m_name, strat), v in stats.items():
        recov = (v['ret'] / v['bet'] * 100) if v['bet'] > 0 else 0.0
        profit = v['ret'] - v['bet']
        hit_rate = (v['hits'] / v['races'] * 100) if v['races'] > 0 else 0.0
        results_list.append({
            'model': m_name, 'strategy': strat,
            'recovery': recov, 'profit': profit,
            'bet': v['bet'], 'ret': v['ret'],
            'hit_rate': hit_rate, 'races': v['races'], 'hits': v['hits']
        })

    df_results = pd.DataFrame(results_list).sort_values('recovery', ascending=False)

    report_path = os.path.join(REPORT_DIR, f'simulation_all_models_2025_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 2025年 全モデル/戦略シミュレーション結果\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 予算: {BUDGET}円/レース\n- 対象期間: 2025年全期間\n- 手法: 前処理済みデータ使用（高速版）\n\n")
        f.write("## ランキング（回収率順）\n\n")
        f.write("| 順位 | モデル | 戦略 | 回収率 | 収益 | 投資額 | 的中率 | 対象レース |\n")
        f.write("|------|--------|--------|--------|------|------|--------|--------|\n")
        for i, row in enumerate(df_results.itertuples(), 1):
            f.write(f"| {i} | {row.model} | {row.strategy} | {row.recovery:.1f}% | {row.profit:+,}円 | {row.bet:,}円 | {row.hit_rate:.1f}% | {row.races}R |\n")

        best = df_results.iloc[0]
        f.write(f"\n## 最優秀組み合わせ\n\n")
        f.write(f"**モデル: `{best.model}` / 戦略: `{best.strategy}`**\n\n")
        f.write(f"- 回収率: {best.recovery:.1f}%\n- 収益: {best.profit:+,}円\n- 的中率: {best.hit_rate:.1f}%\n")

    print(f"\n=== シミュレーション完了 ===")
    print(f"レポート: {report_path}")
    print(f"\n--- 結果ランキング（回収率順）---")
    print(df_results[['model', 'strategy', 'recovery', 'profit', 'hit_rate']].to_string(index=False))

    best = df_results.iloc[0]
    print(f"\n★ 最優秀: model={best.model}, strategy={best.strategy}, recovery={best.recovery:.1f}%")

    result_json = os.path.join(REPORT_DIR, 'best_model_strategy.json')
    with open(result_json, 'w', encoding='utf-8') as fj:
        json.dump({
            'model': best.model,
            'strategy': best.strategy,
            'recovery': float(best.recovery),
            'profit': int(best.profit),
            'hit_rate': float(best.hit_rate)
        }, fj, ensure_ascii=False, indent=2)

    print(f"最優秀組み合わせをJSON保存: {result_json}")


if __name__ == '__main__':
    main()
