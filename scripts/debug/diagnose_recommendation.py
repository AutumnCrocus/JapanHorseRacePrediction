"""
推奨買い目生成プロセスの診断スクリプト
存在しない馬番12が推奨される原因を特定する
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.data_loader import fetch_and_process_race_data
from modules.scraping import Odds
from modules.strategy import BettingStrategy
import pandas as pd

# テスト対象のレースID
RACE_ID = "202610010111"

print("=" * 80)
print(f"診断開始: レースID {RACE_ID}")
print("=" * 80)

# ステップ1: 出馬データの取得
print("\n[ステップ1] 出馬データの取得")
try:
    df = fetch_and_process_race_data(RACE_ID)
    print(f"[OK] 出馬データ取得成功: {len(df)}頭")
    print(f"  馬番の範囲: {df['馬番'].min()} - {df['馬番'].max()}")
    print(f"  馬番リスト: {sorted(df['馬番'].unique().tolist())}")
except Exception as e:
    print(f"[NG] エラー: {e}")
    sys.exit(1)

# ステップ2: 予測結果の模擬（AIモデルを使わず、ダミー確率を使用）
print("\n[ステップ2] 予測結果の作成（ダミーデータ）")
predictions = []
for _, row in df.iterrows():
    horse_no = int(row['馬番'])
    predictions.append({
        'horse_number': horse_no,
        'horse_name': row['馬名'],
        'probability': 0.5,  # ダミー確率
        'odds': row.get('単勝', 10.0),
        'popularity': row.get('人気', 0)
    })

print(f"[OK] 予測結果作成完了: {len(predictions)}頭")
print(f"  予測に含まれる馬番: {sorted([p['horse_number'] for p in predictions])}")

# ステップ3: オッズデータの取得
print("\n[ステップ3] オッズデータの取得")
try:
    odds_data = Odds.scrape(RACE_ID)
    print(f"[OK] オッズデータ取得完了")
    print(f"  単勝オッズ: {len(odds_data['tan'])}件")
    print(f"    馬番: {sorted(odds_data['tan'].keys())}")
    print(f"  複勝オッズ: {len(odds_data['fuku'])}件")
    print(f"    馬番: {sorted(odds_data['fuku'].keys())}")
except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ステップ4: 推奨買い目の生成
print("\n[ステップ4] 推奨買い目の生成")
try:
    df_preds = pd.DataFrame(predictions)
    rec_df = BettingStrategy.calculate_expected_value(df_preds, odds_data)
    
    print(f"[OK] 推奨買い目計算完了: {len(rec_df)}件")
    
    if not rec_df.empty:
        print("\n  推奨内容:")
        for _, rec in rec_df.iterrows():
            print(f"    - 券種: {rec['type']}, 馬番: {rec['umaban']}, 馬名: {rec['name']}, EV: {rec['ev']:.2f}")
            
        # 馬番12が含まれているかチェック
        print("\n[診断結果]")
        problematic_recs = rec_df[rec_df['umaban'].astype(str).str.contains('12')]
        
        if len(problematic_recs) > 0:
            print(f"[警告] 馬番12を含む推奨が{len(problematic_recs)}件見つかりました:")
            for _, rec in problematic_recs.iterrows():
                print(f"  - 券種: {rec['type']}, 馬番: {rec['umaban']}, 馬名: {rec['name']}")
                print(f"    オッズ: {rec['odds']}, 確率: {rec['prob']:.3f}, EV: {rec['ev']:.2f}")
        else:
            print("[OK] 馬番12を含む推奨は見つかりませんでした。")
    else:
        print("  推奨買い目なし（EVが閾値未満）")
        
except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ステップ5: 予算配分
print("\n[ステップ5] 予算配分（budget=1000円）")
try:
    budget = 1000
    rec_list = BettingStrategy.optimize_allocation(rec_df, budget)
    
    print(f"[OK] 予算配分完了: {len(rec_list)}件")
    
    if rec_list:
        print("\n  配分結果:")
        for i, rec in enumerate(rec_list, 1):
            print(f"    {i}. 券種: {rec['type']}, 馬番: {rec['umaban']}, 金額: ¥{rec['amount']}")
            
        # 馬番12が含まれているかチェック
        print("\n[最終診断結果]")
        problematic_final = [r for r in rec_list if '12' in str(r['umaban'])]
        
        if problematic_final:
            print(f"[警告] 最終的な推奨リストに馬番12を含む買い目が{len(problematic_final)}件含まれています:")
            for rec in problematic_final:
                print(f"  - 券種: {rec['type']}, 馬番: {rec['umaban']}, 金額: ¥{rec['amount']}")
                print(f"    馬名: {rec['name']}, オッズ: {rec['odds']}, EV: {rec['ev']:.2f}")
        else:
            print("[OK] 最終的な推奨リストに馬番12は含まれていません。")
    else:
        print("  配分結果なし")
        
except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("診断完了")
print("=" * 80)
