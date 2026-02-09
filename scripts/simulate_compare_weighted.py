"""
時系列重み付けモデル vs ヒストリカルモデル 比較シミュレーション (改善版)
- 標準出力: 結果サマリーのみ
- 詳細ログ: logs/ へファイル出力
- サンプルモード: --sample で10%のみ実行
- 進捗追跡: simulation_progress.json
"""
import os
import sys
import json
import pickle
import random
import argparse
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

HISTORICAL_MODEL_DIR = os.path.join(MODEL_DIR, "historical_2010_2024")
WEIGHTED_MODEL_DIR = os.path.join(MODEL_DIR, "weighted_2010_2024")

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), '..', 'simulation_progress.json')

os.makedirs(LOGS_DIR, exist_ok=True)


class SimulationLogger:
    """ファイルベースのロガー"""
    def __init__(self, filename: str):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.start_time = datetime.now()
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Simulation Log Started: {self.start_time.isoformat()} ===\n")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def error(self, message: str):
        """ERROR prefix for grep"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: {message}\n")


def update_progress(status: str, model: str, processed: int, total: int, 
                    sample_mode: bool, summary: dict):
    """進捗ファイルを更新"""
    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "current_model": model,
        "progress_pct": round(processed / total * 100, 1) if total > 0 else 0,
        "races_processed": processed,
        "races_total": total,
        "sample_mode": sample_mode,
        "summary": summary
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


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
    
    # 2025年抽出
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


def load_model_and_aux(model_dir: str):
    """モデルと補助オブジェクトを読み込み"""
    model = HorseRaceModel()
    model.load(os.path.join(model_dir, 'model.pkl'))
    
    with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
        processor = pickle.load(f)
    with open(os.path.join(model_dir, 'engineer.pkl'), 'rb') as f:
        engineer = pickle.load(f)
    
    return RacePredictor(model, processor, engineer)


def process_race(race_df, predictor, hr, peds, logger):
    """単一レースの特徴量構築"""
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
    except Exception as e:
        logger.error(f"process_race failed: {str(e)}")
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


def verify_hit(race_id, rec, returns_dict, logger):
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

            except Exception:
                pass
    except Exception as e:
        logger.error(f"verify_hit failed for {race_id}: {str(e)}")
    
    return payout


def simulate_model(model_name: str, predictor, hr, peds, df_target, returns_dict, 
                   sample_mode: bool, logger):
    """単一モデルでシミュレーション実行"""
    CONFIDENCE_ORDER = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    
    race_ids = df_target['race_id'].unique().tolist()
    total_races = len(race_ids)
    
    # サンプルモード
    if sample_mode:
        sample_size = max(1, int(total_races * 0.1))
        race_ids = random.sample(race_ids, sample_size)
        logger.log(f"Sample mode: {sample_size}/{total_races} races selected")
    
    total_bet = 0
    total_return = 0
    race_count = 0
    bet_count = 0
    hit_count = 0
    
    update_interval = max(1, len(race_ids) // 10)  # 10%刻み
    
    for i, race_id in enumerate(race_ids):
        try:
            race_df = df_target[df_target['race_id'] == race_id].copy()
            if race_df.empty:
                continue
            
            df_pred = process_race(race_df, predictor, hr, peds, logger)
            if df_pred is None or df_pred.empty:
                continue
            
            conf = calculate_confidence(df_pred)
            
            if CONFIDENCE_ORDER.get(conf, 99) > CONFIDENCE_ORDER.get(MIN_CONFIDENCE, 1):
                continue
            
            race_count += 1
            
            recs = BettingAllocator.allocate_budget(df_pred, BUDGET, strategy=STRATEGY)
            if not recs:
                continue
            
            for rec in recs:
                bet_amount = rec.get('total_amount', rec.get('amount', 100))
                total_bet += bet_amount
                bet_count += 1
                
                ret = verify_hit(race_id, rec, returns_dict, logger)
                total_return += ret
                if ret > 0:
                    hit_count += 1
                    logger.log(f"HIT: {race_id} - ¥{int(ret):,}")
            
            # 進捗更新 (10%刻み)
            if (i + 1) % update_interval == 0:
                summary = {
                    "bets": total_bet,
                    "returns": total_return,
                    "hit_count": hit_count,
                    "bet_count": bet_count
                }
                update_progress("running", model_name, i + 1, len(race_ids), sample_mode, summary)
                    
        except Exception as e:
            logger.error(f"Race {race_id} failed: {str(e)}")
            continue
    
    recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0
    hit_rate = (hit_count / bet_count * 100) if bet_count > 0 else 0
    
    return {
        'model': model_name,
        'race_count': race_count,
        'bet_count': bet_count,
        'hit_count': hit_count,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'recovery_rate': recovery_rate,
        'profit': total_return - total_bet,
        'sample_mode': sample_mode
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Run 10% sample only')
    parser.add_argument('--full', action='store_true', help='Run full simulation')
    args = parser.parse_args()
    
    sample_mode = not args.full  # デフォルトはサンプルモード
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimulationLogger(f"simulation_{timestamp}.log")
    
    logger.log(f"Strategy: {STRATEGY}, Budget: {BUDGET}, Min Confidence: {MIN_CONFIDENCE}")
    logger.log(f"Sample Mode: {sample_mode}")
    
    # 進捗初期化
    update_progress("loading", "", 0, 0, sample_mode, {})
    
    # データ読み込み
    logger.log("Loading data...")
    df_target, hr, peds, returns_dict = load_resources()
    total_races = df_target['race_id'].nunique()
    logger.log(f"Target races: {total_races}")
    
    # モデル読み込み
    logger.log("Loading models...")
    hist_predictor = load_model_and_aux(HISTORICAL_MODEL_DIR)
    weight_predictor = load_model_and_aux(WEIGHTED_MODEL_DIR)
    
    # シミュレーション実行
    logger.log("=== ヒストリカルモデル シミュレーション開始 ===")
    results_hist = simulate_model("historical", hist_predictor, hr, peds, df_target, returns_dict, sample_mode, logger)
    logger.log(f"Historical: Recovery={results_hist['recovery_rate']:.1f}%, Profit=¥{results_hist['profit']:,}")
    
    logger.log("=== 時系列重み付けモデル シミュレーション開始 ===")
    results_weight = simulate_model("weighted", weight_predictor, hr, peds, df_target, returns_dict, sample_mode, logger)
    logger.log(f"Weighted: Recovery={results_weight['recovery_rate']:.1f}%, Profit=¥{results_weight['profit']:,}")
    
    # 最終進捗
    update_progress("completed", "both", total_races, total_races, sample_mode, {
        "historical": results_hist,
        "weighted": results_weight
    })
    
    # === 標準出力: 結果サマリーのみ ===
    mode_str = "(10% Sample)" if sample_mode else "(Full)"
    print(f"\n{'='*50}")
    print(f"Simulation Complete {mode_str}")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Recovery':>10} {'Profit':>15}")
    print(f"{'-'*50}")
    print(f"{'Historical':<20} {results_hist['recovery_rate']:>9.1f}% ¥{results_hist['profit']:>13,}")
    print(f"{'Weighted':<20} {results_weight['recovery_rate']:>9.1f}% ¥{results_weight['profit']:>13,}")
    print(f"{'-'*50}")
    
    diff = results_weight['recovery_rate'] - results_hist['recovery_rate']
    if diff > 0:
        print(f"✓ Weighted model +{diff:.1f}% better")
    else:
        print(f"✗ Historical model +{abs(diff):.1f}% better")
    
    print(f"\nLog: {logger.filepath}")
    print(f"Progress: {PROGRESS_FILE}")
    
    # レポート保存
    report_path = os.path.join(os.path.dirname(__file__), '..', 'comparison_report_weighted_vs_historical.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 時系列重み付けモデル vs ヒストリカルモデル 比較レポート\n\n")
        f.write(f"- **実行日時**: {datetime.now().isoformat()}\n")
        f.write(f"- **モード**: {'10% サンプル' if sample_mode else 'フル実行'}\n")
        f.write(f"- **戦略**: {STRATEGY}\n")
        f.write(f"- **予算**: ¥{BUDGET}\n")
        f.write(f"- **自信度フィルタ**: {MIN_CONFIDENCE}以上\n\n")
        f.write("## 結果サマリー\n\n")
        f.write("| モデル | 回収率 | 的中率 | 収支 |\n")
        f.write("|--------|--------|--------|------|\n")
        f.write(f"| ヒストリカル | {results_hist['recovery_rate']:.1f}% | {results_hist['hit_rate']:.1f}% | ¥{results_hist['profit']:,} |\n")
        f.write(f"| 時系列重み付け | {results_weight['recovery_rate']:.1f}% | {results_weight['hit_rate']:.1f}% | ¥{results_weight['profit']:,} |\n\n")
        
        if diff > 0:
            f.write(f"**結論**: 時系列重み付けモデルが **{diff:.1f}%** 上回りました。\n")
        else:
            f.write(f"**結論**: ヒストリカルモデルの方が **{abs(diff):.1f}%** 優れています。\n")


if __name__ == "__main__":
    main()
