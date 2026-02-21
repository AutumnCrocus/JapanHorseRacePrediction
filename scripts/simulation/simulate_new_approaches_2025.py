"""
2025年 全アプローチ比較シミュレーション
- lgbm (現行ベスト)
- ltr
- stacking
- catboost (新規)
ベース戦略: box4_sanrenpuku
EVフィルタあり (EV>=2.5): 最良フィルタ
EVフィルタなし: ベースライン
予算: 5000円/レース
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

BUDGET = 5000
BASE_STRATEGY = 'box4_sanrenpuku'
EV_THRESHOLD = 2.5  # 最良フィルタ
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

BET_KEY_MAP = {
    '単勝': 'tan', '複勝': 'fuku', '馬連': 'umaren', 'ワイド': 'wide',
    '馬単': 'umatan', '3連複': 'sanrenpuku', '3連単': 'sanrentan'
}


class LTRWrapper:
    def __init__(self, data_dict: dict):
        self._model = data_dict['model']
        self.feature_names = data_dict['feature_names']
        self.model_type = 'ltr'

    def predict(self, X):
        return self._model.predict(X[self.feature_names])


class CatBoostWrapper:
    """CatBoostモデルラッパー (HorseRaceModel 互換)"""
    def __init__(self, model_data: dict):
        self._model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = 'catboost'

    def predict(self, X):
        try:
            import catboost as cb
            X_aligned = X[self.feature_names].fillna(0)
            return self._model.predict_proba(X_aligned)[:, 1]
        except Exception as e:
            print(f"CatBoost予測エラー: {e}")
            return np.zeros(len(X))


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


def load_data():
    print("[1] データ読み込み中...", flush=True)
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    with open(dataset_path, 'rb') as f:
        ds = pickle.load(f)

    df = ds['data']
    lgbm_features = ds.get('feature_names', [])

    if 'original_race_id' in df.columns:
        df['race_id'] = df['original_race_id'].astype(str)
    elif 'race_id' not in df.columns:
        df['race_id'] = df.index.astype(str)

    df_2025 = df[df['year'] == 2025].copy() if 'year' in df.columns else df.copy()
    print(f"   2025年データ: {len(df_2025)}行, レース数: {len(df_2025['race_id'].unique())}", flush=True)

    print("[2] 配当データ読み込み中...", flush=True)
    payouts = load_payouts_2025()
    print(f"   配当データ: {len(payouts)}レース", flush=True)

    return df_2025, lgbm_features, payouts


def load_models():
    print("[3] モデルロード中...", flush=True)
    models = {}

    # LGBM
    try:
        m = HorseRaceModel()
        m.load(os.path.join(MODEL_DIR, 'historical_2010_2024', 'model.pkl'))
        models['lgbm'] = ('lgbm', m)
        print("   [lgbm] ロード完了", flush=True)
    except Exception as e:
        print(f"   [lgbm] ロード失敗: {e}", flush=True)

    # LTR
    try:
        ltr_path = os.path.join(MODEL_DIR, 'standalone_ranking', 'ranking_model.pkl')
        with open(ltr_path, 'rb') as f:
            ltr_data = pickle.load(f)
        models['ltr'] = ('ltr', LTRWrapper(ltr_data))
        print("   [ltr] ロード完了", flush=True)
    except Exception as e:
        print(f"   [ltr] ロード失敗: {e}", flush=True)

    # Stacking
    try:
        m_st = HorseRaceModel()
        m_st.load(os.path.join(MODEL_DIR, 'experiment_model_2026.pkl'))
        models['stacking'] = ('stacking', m_st)
        print("   [stacking] ロード完了", flush=True)
    except Exception as e:
        print(f"   [stacking] ロード失敗: {e}", flush=True)

    # CatBoost (新規)
    try:
        cat_path = os.path.join(MODEL_DIR, 'catboost_2010_2024', 'model.pkl')
        with open(cat_path, 'rb') as f:
            cat_data = pickle.load(f)
        models['catboost'] = ('catboost', CatBoostWrapper(cat_data))
        print("   [catboost] ロード完了", flush=True)
    except Exception as e:
        print(f"   [catboost] ロード失敗: {e}", flush=True)

    return models


def predict_race(race_df, model_type, model, feature_names):
    try:
        X = pd.DataFrame(index=race_df.index)
        for c in feature_names:
            X[c] = pd.to_numeric(race_df[c], errors='coerce').fillna(0) if c in race_df.columns else 0.0

        if 'deepfm_score' in feature_names and 'deepfm_score' not in race_df.columns:
            X['deepfm_score'] = 0.5

        probs = model.predict(X)

        if model_type == 'ltr':
            mn, mx = probs.min(), probs.max()
            probs = (probs - mn) / (mx - mn) if mx != mn else np.full(len(probs), 0.5)

        df_res = race_df.copy()
        df_res['probability'] = probs
        df_res['horse_number'] = pd.to_numeric(
            df_res.get('馬番', pd.Series(range(1, len(df_res) + 1))), errors='coerce'
        ).fillna(0).astype(int)
        df_res['odds'] = pd.to_numeric(
            df_res.get('単勝', pd.Series([10.0] * len(df_res))), errors='coerce'
        ).fillna(10.0)
        df_res['expected_value'] = df_res['probability'] * df_res['odds']
        return df_res
    except Exception:
        return None


def compute_race_ev(df_preds):
    return float(df_preds.sort_values('probability', ascending=False)['expected_value'].head(4).mean())


def verify_hit(rec, race_pay):
    bet_key = BET_KEY_MAP.get(rec.get('bet_type', ''))
    if not bet_key or bet_key not in race_pay:
        return 0

    winning_data = race_pay[bet_key]
    method = rec.get('method', '')
    bet_horses = [int(h) for h in rec.get('horse_numbers', []) if str(h).isdigit()]
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


def run_simulation(m_name, model_type, model, feature_names, race_ids, df_2025, payouts, ev_filter=False):
    stats = {'bet': 0, 'ret': 0, 'hits': 0, 'races': 0, 'skipped': 0}

    for race_id in race_ids:
        race_df = df_2025[df_2025['race_id'] == race_id]
        if len(race_df) < 4:
            continue

        df_preds = predict_race(race_df, model_type, model, feature_names)
        if df_preds is None or df_preds.empty:
            continue

        df_sorted = df_preds.sort_values('probability', ascending=False)

        if ev_filter:
            race_ev = compute_race_ev(df_sorted)
            if race_ev < EV_THRESHOLD:
                stats['skipped'] += 1
                continue

        race_pay = payouts.get(str(race_id))

        try:
            recs = BettingAllocator.allocate_budget(df_sorted, BUDGET, strategy=BASE_STRATEGY)
        except Exception:
            continue

        if not recs:
            continue

        stats['races'] += 1
        race_hit = False
        for rec in recs:
            amt = rec.get('total_amount', 0)
            stats['bet'] += amt
            if race_pay:
                pay = verify_hit(rec, race_pay)
                stats['ret'] += pay
                if pay > 0:
                    race_hit = True
        if race_hit:
            stats['hits'] += 1

    recov = (stats['ret'] / stats['bet'] * 100) if stats['bet'] > 0 else 0.0
    profit = stats['ret'] - stats['bet']
    hit_rate = (stats['hits'] / stats['races'] * 100) if stats['races'] > 0 else 0.0

    return {
        'model': m_name,
        'ev_filter': f'EV>={EV_THRESHOLD}' if ev_filter else '（なし）',
        'recovery': recov,
        'profit': profit,
        'bet': stats['bet'],
        'hit_rate': hit_rate,
        'races': stats['races'],
        'hits': stats['hits'],
        'skipped': stats['skipped']
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"=== 全アプローチ比較シミュレーション開始: {timestamp} ===\n", flush=True)

    df_2025, lgbm_features, payouts = load_data()
    models = load_models()

    if not models:
        print("ERROR: モデルを1つも読み込めませんでした。")
        return

    race_ids = sorted(df_2025['race_id'].unique().tolist())
    print(f"\n対象レース: {len(race_ids)}件, 戦略: {BASE_STRATEGY}, EV閾値: {EV_THRESHOLD}\n", flush=True)

    results = []
    total = len(models) * 2
    count = 0

    for m_name, (model_type, model) in models.items():
        feat = model.feature_names if hasattr(model, 'feature_names') else lgbm_features

        # EVフィルタなし
        count += 1
        print(f"[{count}/{total}] {m_name} × フィルタなし...", flush=True)
        res = run_simulation(m_name, model_type, model, feat, race_ids, df_2025, payouts, ev_filter=False)
        results.append(res)
        print(f"   → 回収率: {res['recovery']:.1f}%, 収益: {res['profit']:+,}円 ({res['races']}R)", flush=True)

        # EVフィルタあり
        count += 1
        print(f"[{count}/{total}] {m_name} × EV>={EV_THRESHOLD}...", flush=True)
        res = run_simulation(m_name, model_type, model, feat, race_ids, df_2025, payouts, ev_filter=True)
        results.append(res)
        print(f"   → 回収率: {res['recovery']:.1f}%, 収益: {res['profit']:+,}円 ({res['races']}R, skip={res['skipped']}R)", flush=True)

    df_results = pd.DataFrame(results).sort_values('recovery', ascending=False)

    report_path = os.path.join(REPORT_DIR, f'comparison_new_approaches_2025_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 新アプローチ比較シミュレーション結果 (2025年)\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- ベース戦略: `{BASE_STRATEGY}`, 予算: {BUDGET}円/レース\n")
        f.write(f"- EV = 上位4頭の（予測確率 × 単勝オッズ）の平均\n\n")
        f.write("## 全組み合わせランキング（回収率順）\n\n")
        f.write("| 順位 | モデル | EVフィルタ | 回収率 | 収益 | 的中率 | 投票レース |\n")
        f.write("|------|--------|-----------|--------|------|--------|----------|\n")

        for i, row in enumerate(df_results.itertuples(), 1):
            f.write(
                f"| {i} | {row.model} | {row.ev_filter} "
                f"| **{row.recovery:.1f}%** | {int(row.profit):+,}円 "
                f"| {row.hit_rate:.1f}% | {int(row.races)}R |\n"
            )

        best = df_results.iloc[0]
        f.write(f"\n## 最優秀組み合わせ\n\n")
        f.write(f"**モデル: `{best.model}` / EVフィルタ: `{best.ev_filter}`**\n\n")
        f.write(f"- 回収率: {best.recovery:.1f}%\n")
        f.write(f"- 収益: {int(best.profit):+,}円\n")
        f.write(f"- 的中率: {best.hit_rate:.1f}%\n")
        f.write(f"- 投票レース: {int(best.races)}R\n\n")

        # ベースライン比較
        baseline = df_results[(df_results['model'] == 'lgbm') & (df_results['ev_filter'] == '（なし）')]
        if not baseline.empty:
            bl = baseline.iloc[0]
            f.write(f"## ベースライン比較 (lgbm / フィルタなし)\n\n")
            f.write(f"- 回収率: {bl.recovery:.1f}%\n")
            f.write(f"- 収益: {int(bl.profit):+,}円\n")
            f.write(f"- 改善率: {best.recovery / bl.recovery * 100 - 100:.1f}%アップ\n")

    print(f"\n=== シミュレーション完了 ===")
    print(f"レポート: {report_path}")
    print(df_results[['model', 'ev_filter', 'recovery', 'profit', 'hit_rate', 'races']].to_string(index=False))

    # 最優秀をJSONに保存
    best = df_results.iloc[0]
    best_json = {
        'model': best.model,
        'ev_filter': str(best.ev_filter),
        'ev_threshold': EV_THRESHOLD if 'EV' in str(best.ev_filter) else None,
        'strategy': BASE_STRATEGY,
        'recovery': float(best.recovery),
        'profit': int(best.profit),
        'hit_rate': float(best.hit_rate),
        'races': int(best.races)
    }
    with open(os.path.join(REPORT_DIR, 'best_model_strategy_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    print(f"\n★ 最優秀: {best.model} / {best.ev_filter} → 回収率 {best.recovery:.1f}%")


if __name__ == '__main__':
    main()
