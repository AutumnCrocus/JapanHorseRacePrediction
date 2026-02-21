"""
本日のレース予想 (2026-02-21)
モデル: CatBoost (2010-2024学習)
戦略: box4_sanrenpuku
フィルタ: EV >= 2.5 (上位4頭平均期待値)
対象: 6R以降
"""
import os
import sys
import pickle
import pandas as pd
import time
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import app
from app import load_model, MODELS, PROCESSORS, ENGINEERS
from modules.constants import MODEL_DIR
from modules.data_loader import fetch_and_process_race_data
from modules.betting_allocator import BettingAllocator

TARGET_DATE = "20260221"
BUDGET_PER_RACE = 5000
EV_THRESHOLD = 2.5
BASE_STRATEGY = 'box4_sanrenpuku'

class CatBoostWrapper:
    """CatBoostモデルラッパー (HorseRaceModel 互換)"""
    def __init__(self, model_data: dict):
        self._model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = 'catboost'

    def predict(self, X):
        import catboost as cb
        # 特徴量の順序を合わせる
        X_aligned = X[self.feature_names].fillna(0)
        return self._model.predict_proba(X_aligned)[:, 1]

def compute_race_ev(df_preds: pd.DataFrame) -> float:
    """上位4頭の平均期待値を算出"""
    df_sorted = df_preds.sort_values('probability', ascending=False)
    top4_ev = df_sorted['expected_value'].head(4)
    if top4_ev.empty:
        return 0.0
    return float(top4_ev.mean())

def main():
    print(f"=== {TARGET_DATE} レース予想開始 (CatBoost × EV>=2.5) ===")
    
    # 1. マスターデータとモデルのロード
    print("マスターデータをロード中...")
    load_model('lgbm') # 共通データのロードを兼ねる
    
    cat_path = os.path.join(MODEL_DIR, 'catboost_2010_2024', 'model.pkl')
    if not os.path.exists(cat_path):
        print(f"Error: CatBoostモデルが見つかりません: {cat_path}")
        return
        
    with open(cat_path, 'rb') as f:
        cat_data = pickle.load(f)
    model = CatBoostWrapper(cat_data)
    print("CatBoostモデル ロード完了")

    # 2. 対象レースIDの取得
    # 東京(05): 2026050107, 阪神(09): 2026090101, 小倉(10): 2026100107
    race_ids = []
    venues_config = [
        ("050107", range(6, 13)), # 東京 6-12R
        ("090101", range(6, 13)), # 阪神 6-12R
        ("100107", range(6, 13)), # 小倉 6-12R
    ]
    for prefix, r_range in venues_config:
        for r_num in r_range:
            race_ids.append(f"2026{prefix}{r_num:02d}")
    
    race_ids = sorted(race_ids)
    
    if not race_ids:
        print("対象レースが見つかりませんでした。")
        return
    
    print(f"対象レース数 (6R以降): {len(race_ids)}件")

    results = []
    
    # 3. 予測実行ループ
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
            race_name = df.attrs.get('race_name', '不明')
            race_data01 = df.attrs.get('race_data01', '')
            race_data02 = df.attrs.get('race_data02', '')
            print(f"レース名: {race_name} ({race_data01})")

            # 特徴量抽出
            X = pd.DataFrame(index=df.index)
            for c in model.feature_names:
                 X[c] = pd.to_numeric(df[c], errors='coerce').fillna(0) if c in df.columns else 0.0
            
            # 予測
            probs = model.predict(X)
            
            df_preds = df.copy()
            df_preds['probability'] = probs
            df_preds['horse_number'] = pd.to_numeric(df_preds.get('馬番', df_preds.index+1), errors='coerce').fillna(0).astype(int)
            df_preds['odds'] = pd.to_numeric(df_preds.get('単勝', 10.0), errors='coerce').fillna(10.0)
            df_preds['expected_value'] = df_preds['probability'] * df_preds['odds']
            
            # EV算出
            race_ev = compute_race_ev(df_preds)
            print(f"レースEV: {race_ev:.2f}")

            # フィルタリング
            if race_ev < EV_THRESHOLD:
                print(f"Skip {rid}: EV {race_ev:.2f} < 閾値 {EV_THRESHOLD}")
                continue
            
            print(f"★ 投票対象レース (EV {race_ev:.2f} >= {EV_THRESHOLD})")
            
            # 買い目生成
            with test_app.app_context():
                recs = BettingAllocator.allocate_budget(df_preds.sort_values('probability', ascending=False), 
                                                       BUDGET_PER_RACE, strategy=BASE_STRATEGY)
            
            if recs:
                results.append({
                    'race_id': rid,
                    'race_name': race_name,
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

    # 4. レポート生成
    report_file = f"reports/prediction_{TARGET_DATE}_catboost_ev_{datetime.now().strftime('%H%M%S')}.md"
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 本日のレース予想レポート ({TARGET_DATE})\n\n")
        f.write(f"- 実行時刻: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")
        f.write(f"- モデル: `CatBoost` (2024年まで学習)\n")
        f.write(f"- フィルタ: **EV >= {EV_THRESHOLD}** (上位4頭平均期待値)\n")
        f.write(f"- 戦略: `{BASE_STRATEGY}` / 予算: {BUDGET_PER_RACE}円/レース\n\n")
        
        if not results:
            f.write("## 投票対象となる高期待値レースは見つかりませんでした。\n")
        else:
            f.write(f"## 投票候補レース一覧 ({len(results)}件)\n\n")
            for res in results:
                f.write(f"### {res['race_name']} ({res['race_id']})\n")
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
