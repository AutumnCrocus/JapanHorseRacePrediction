import os
import sys
import pandas as pd
from datetime import datetime

# ============= 設定 =============
if len(sys.argv) > 1:
    TARGET_DATE = sys.argv[1]
else:
    TARGET_DATE = '20260222'

DAILY_FILE = f"reports/verify_integrated_result_{TARGET_DATE}.csv"
CUMULATIVE_FILE = "reports/cumulative_model_performance.csv"
# ===============================

def main():
    if not os.path.exists(DAILY_FILE):
        print(f"Error: {DAILY_FILE} not found.")
        sys.exit(1)

    print(f"Reading daily performance from: {DAILY_FILE}")
    df_daily = pd.read_csv(DAILY_FILE)
    df_daily['Date'] = str(TARGET_DATE)

    # 必要な列がなければ作成
    if not os.path.exists(CUMULATIVE_FILE):
        print(f"Creating new cumulative file: {CUMULATIVE_FILE}")
        df_cumulative = pd.DataFrame()
    else:
        print(f"Loading existing cumulative file: {CUMULATIVE_FILE}")
        df_cumulative = pd.read_csv(CUMULATIVE_FILE)
        if 'Date' in df_cumulative.columns:
            df_cumulative['Date'] = df_cumulative['Date'].astype(str)

    # 既存のターゲット日付のデータがあれば削除（重複防止）
    if not df_cumulative.empty and 'Date' in df_cumulative.columns:
        df_cumulative = df_cumulative[df_cumulative['Date'] != str(TARGET_DATE)]

    # 追加
    df_cumulative = pd.concat([df_cumulative, df_daily], ignore_index=True)

    # 保存
    os.makedirs(os.path.dirname(CUMULATIVE_FILE), exist_ok=True)
    df_cumulative.to_csv(CUMULATIVE_FILE, index=False, encoding='utf-8-sig')
    print(f"Successfully saved cumulative performance to: {CUMULATIVE_FILE}")

    # ===== モデルごとの累積サマリー出力 =====
    # Dateごとの合算ではなく、StrategyでGroupByして累積の回収率等を再計算
    summary = []
    grouped = df_cumulative.groupby('Strategy')
    for strat, group in grouped:
        total_bet = group['総投資(円)'].sum()
        total_ret = group['回収額(円)'].sum()
        total_hits = group['的中回数'].sum()
        total_counts = group['総ベット数(点)'].sum()
        net = total_ret - total_bet
        roi = (total_ret / total_bet * 100) if total_bet > 0 else 0
        hit_rate = (total_hits / total_counts * 100) if total_counts > 0 else 0
        
        summary.append({
            'Strategy': strat,
            '累積_総投資(円)': total_bet,
            '累積_回収額(円)': int(total_ret),
            '累積_収支(円)': int(net),
            '累積_回収率(%)': round(roi, 1),
            '累積_的中回数': total_hits,
            '累積_総ベット数(点)': total_counts,
            '累積_的中率(%)': round(hit_rate, 2)
        })
    df_summary = pd.DataFrame(summary).sort_values('累積_回収率(%)', ascending=False)
    
    # ターミナルへ出力
    from tabulate import tabulate
    print("\n=== 各モデル・戦略の累積成績 ===")
    print(tabulate(df_summary, headers='keys', tablefmt='pipe', showindex=False))

    summary_file = "reports/cumulative_model_summary.csv"
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved cumulative summary to: {summary_file}")

if __name__ == "__main__":
    main()
