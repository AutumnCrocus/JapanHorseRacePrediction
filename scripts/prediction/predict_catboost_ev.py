import os
import sys
import pickle
import pandas as pd
import time
import argparse
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import app
from app import load_model, MODELS, PROCESSORS, ENGINEERS
from modules.constants import MODEL_DIR
from modules.data_loader import fetch_and_process_race_data
from modules.betting_allocator import BettingAllocator
from modules.scraping import Shutuba, get_race_date_info

class CatBoostWrapper:
    """CatBoostモデルラッパー"""
    def __init__(self, model_data: dict):
        self._model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = 'catboost'

    def predict(self, X):
        import catboost as cb
        X_aligned = X[self.feature_names].fillna(0)
        return self._model.predict_proba(X_aligned)[:, 1]

def compute_race_ev(df_preds: pd.DataFrame) -> float:
    """上位4頭の平均期待値を算出"""
    df_sorted = df_preds.sort_values('probability', ascending=False)
    top4_ev = df_sorted['expected_value'].head(4)
    if top4_ev.empty:
        return 0.0
    return float(top4_ev.mean())

def discover_race_ids(target_date):
    print(f"{target_date} のレースIDを探索中...")
    venues = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"] 
    valid_ids = []
    
    for v in venues:
        for kai in range(1, 6):
            for day in range(1, 13):
                race_id_prefix = f"{target_date[:4]}{v}{kai:02d}{day:02d}"
                test_race_id = f"{race_id_prefix}11"
                try:
                    info = Shutuba.scrape(test_race_id)
                    if not info.empty:
                        date_info = get_race_date_info(test_race_id)
                        target_str = f"{target_date[:4]}年{int(target_date[4:6])}月{int(target_date[6:8])}日"
                        if info.attrs.get('race_name') and target_str in date_info.get('date', ''):
                            for r_num in range(1, 13):
                                valid_ids.append(f"{race_id_prefix}{r_num:02d}")
                            print(f"会場 {v} (第{kai}回{day}日目) を発見しました。")
                except:
                    continue
    return valid_ids

def main():
    parser = argparse.ArgumentParser(description="CatBoost × EVフィルタを用いたレース予想スクリプト")
    parser.add_argument("--date", type=str, required=True, help="対象日付 (YYYYMMDD)")
    parser.add_argument("--ev_threshold", type=float, default=2.5, help="EVの閾値 (デフォルト: 2.5)")
    parser.add_argument("--budget", type=int, default=5000, help="1レースあたりの予算 (デフォルト: 5000)")
    parser.add_argument("--strategy", type=str, default="box4_sanrenpuku", help="買い目戦略 (デフォルト: box4_sanrenpuku)")
    args = parser.parse_args()

    TARGET_DATE = args.date
    BUDGET_PER_RACE = args.budget
    EV_THRESHOLD = args.ev_threshold
    BASE_STRATEGY = args.strategy

    print(f"=== {TARGET_DATE} レース予想開始 (CatBoost × EV>={EV_THRESHOLD}) ===")
    
    print("マスターデータをロード中...")
    load_model('lgbm') 
    
    cat_path = os.path.join(MODEL_DIR, 'catboost_2010_2024', 'model.pkl')
    if not os.path.exists(cat_path):
        print(f"Error: CatBoostモデルが見つかりません: {cat_path}")
        return
        
    with open(cat_path, 'rb') as f:
        cat_data = pickle.load(f)
    model = CatBoostWrapper(cat_data)
    print("CatBoostモデル ロード完了")

    all_race_ids = discover_race_ids(TARGET_DATE)
    if not all_race_ids:
        print("有効なレースが見つかりませんでした。")
        return

    # 後半戦（6R以降）に絞る
    race_ids = [rid for rid in all_race_ids if int(rid[-2:]) >= 6]
    
    if not race_ids:
        print("対象レース(6R以降)が見つかりませんでした。")
        return
    
    print(f"対象レース数 (6R以降): {len(race_ids)}件")

    results = []
    from flask import Flask
    test_app = Flask(__name__)
    
    for rid in race_ids:
        df = None
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                print(f"\n>>>> 処理中: {rid} (試行 {attempt+1}) <<<<")
                df = fetch_and_process_race_data(rid, PROCESSORS['lgbm'], ENGINEERS['lgbm'], 
                                                 app.bias_map, app.jockey_stats, 
                                                 app.horse_results, app.peds)
                if df is not None and not df.empty:
                    break
                if attempt < max_retries:
                    print(f"Retry {rid}: 待機中...")
                    time.sleep(2)
            except Exception as e:
                print(f"Fetch Error {rid}: {e}")
                if attempt < max_retries:
                    time.sleep(2)
        
        if df is None or df.empty:
            print(f"Skip {rid}: データ取得失敗")
            continue
            
        try:
            venue_map = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
            venue_code = rid[4:6]
            venue_name = venue_map.get(venue_code, "不明")
            race_num = int(rid[-2:])
            
            race_name = df.attrs.get('race_name', '不明')
            if race_name == '不明':
                race_name = f"{venue_name}{race_num}R"
            
            race_data01 = df.attrs.get('race_data01', '')
            race_data02 = df.attrs.get('race_data02', '')
            print(f"レース名: {race_name} ({venue_name} {race_num}R)")

            X = pd.DataFrame(index=df.index)
            for c in model.feature_names:
                 X[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) if c in df.columns else 0.0
            
            probs = model.predict(X)
            
            df_preds = df.copy()
            df_preds['probability'] = probs
            df_preds['horse_number'] = pd.to_numeric(df_preds.get('馬番', df_preds.index+1), errors='coerce').fillna(0).astype(int)
            df_preds['odds'] = pd.to_numeric(df_preds.get('単勝', 10.0), errors='coerce').fillna(10.0)
            df_preds['expected_value'] = df_preds['probability'] * df_preds['odds']
            
            race_ev = compute_race_ev(df_preds)
            print(f"レースEV: {race_ev:.2f}")

            if race_ev < EV_THRESHOLD:
                print(f"Skip {rid}: EV {race_ev:.2f} < 閾値 {EV_THRESHOLD}")
                continue
            
            print(f"★ 投票対象レース (EV {race_ev:.2f} >= {EV_THRESHOLD})")
            
            with test_app.app_context():
                recs = BettingAllocator.allocate_budget(df_preds.sort_values('probability', ascending=False), 
                                                       BUDGET_PER_RACE, strategy=BASE_STRATEGY)
            
            if recs:
                results.append({
                    'race_id': rid,
                    'race_name': race_name,
                    'venue_name': venue_name,
                    'race_num': race_num,
                    'race_data01': race_data01,
                    'race_data02': race_data02,
                    'race_ev': race_ev,
                    'recs': recs,
                    'predictions': df_preds.sort_values('probability', ascending=False).head(5)
                })
                
        except Exception as e:
            print(f"Error processing {rid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    report_file = f"reports/prediction_{TARGET_DATE}_catboost_ev_{datetime.now().strftime('%H%M%S')}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# {TARGET_DATE[:4]}/{TARGET_DATE[4:6]}/{TARGET_DATE[6:8]} のレース予想レポート\n\n")
        f.write(f"- 実行時刻: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")
        f.write(f"- モデル: `CatBoost` (2024年まで学習)\n")
        f.write(f"- フィルタ: **EV >= {EV_THRESHOLD}** (上位4頭平均期待値)\n")
        f.write(f"- 戦略: `{BASE_STRATEGY}` / 予算: {BUDGET_PER_RACE}円/レース\n\n")
        
        if not results:
            f.write("## 投票対象となる高期待値レースは見つかりませんでした。\n")
        else:
            f.write(f"## 投票候補レース一覧 ({len(results)}件)\n\n")
            for res in results:
                f.write(f"### {res['venue_name']}{res['race_num']}R ({res['race_id']})\n")
                f.write(f"- レース名: {res['race_name']}\n")
                f.write(f"- 開催: {res['race_data01']} / {res['race_data02']}\n")
                f.write(f"- **平均EV: {res['race_ev']:.2f}**\n\n")
                
                f.write("#### 予測上位馬:\n")
                f.write("| 順位 | 馬番 | 馬名 | 予測確率 | オッズ | EV |\n")
                f.write("|:---:|:---:|:---|:---:|:---:|:---:|\n")
                for i, (_, row) in enumerate(res['predictions'].iterrows(), 1):
                    f.write(f"| {i} | {int(row['horse_number'])} | {row.get('馬名', '不明')} | {row['probability']:.1%} | {row['odds']:.1f} | {row['expected_value']:.2f} |\n")
                
                f.write("\n#### 買い目推奨:\n")
                for rec in res['recs']:
                    f.write(f"- **{rec['method']}**: {rec['horse_numbers']} ({rec['total_amount']}円) - {rec['reason']}\n")
                
                f.write("\n---\n\n")
    
    print(f"\n=== 予測完了: {report_file} ===")
    return report_file

if __name__ == "__main__":
    main()
