---
description: 毎週末の運用サイクル（予測実行と成績集計）
---

# 週末の運用サイクル (Weekend Prediction & Evaluation Cycle)

毎週末の全レース予想（予算5000円）と、翌週の各種モデル・戦略の成績集計を行うためのワークフローです。ユーザーの指示（例： `/weekend-cycle`）で開始します。

// turbo-all

## 1. 金曜日・土曜日の夜（翌日レースの予測生成）

翌日のレースに向けたデータ収集（出馬表、オッズなど）と、全モデル・戦略での予測を実施します。

1. **データ収集と前処理**:
   `scripts/data/` 系のスクリプトを実行し、翌日の出馬表などの必要な最新データを取得・整形します。

2. **全モデル・戦略の予測生成（予算5000円）**:
   翌日の日付を指定して予測スクリプトを実行します。各モデル（LGBM, LTR, Stackingなど）と全戦略の買い目を生成します。

   ```bash
   # 例: 2026年2月21日（土）の予測を生成
   python scripts/predict_tomorrow_all_models.py --date 20260221 --budget 5000
   ```

3. **最適な買い目の俺プロ自動登録** (オプション/タスクに応じて):
   前週までのシミュレーション結果から最適なモデル・戦略（例: `lgbm/box4_sanrenpuku`）を選定し、生成された予想を俺プロに登録します。

   ```bash
   python scripts/automation/automate_orepro_best.py
   ```

4. **結果の報告（notify_user）**:
   予測生成が完了した旨と、レポートファイル（`reports/prediction_YYYYMMDD_all_models.md` 等）をユーザーに報告します。

## 2. 週末の夜（日別成績集計と累積ファイルの更新）

レース当日の結果（払戻金など）を取得し、成績の集計と累積ファイル（CSV等）への保存を行います。対象日（`YYYYMMDD`）ごとに以下の手順を実行します。

1. **払戻金データの取得**:
   レース開催日の配当データを取得します。

   ```bash
   python scripts/scraping/scrape_payouts.py [YYYYMMDD]
   ```

2. **当日の成績集計**:
   取得した払戻金データと出力済みの予測データ（例： `prediction_YYYYMMDD_integrated_v2.md`）を突き合わせ、各戦略の回収率などを日次で計算します。

   ```bash
   python scripts/simulation/verify_integrated_report.py [YYYYMMDD]
   ```

3. **成績の累積ファイルへの保存とサマリー更新**:
   当日の成績を過去のデータと統合し、すべてのモデル・戦略の累積成績を更新します。

   ```bash
   python scripts/simulation/update_cumulative_results.py [YYYYMMDD]
   ```

4. **結果の記録と共有**:
   - 出力された各種成績ファイル（`reports/verify_integrated_result_[YYYYMMDD].csv`、`reports/cumulative_model_performance.csv`、`reports/cumulative_model_summary.csv` 等）の内容を確認します。
   - 特に好成績だった、あるいは想定より悪かったモデル・戦略の傾向を分析し、必要に応じて `docs/lessons/` に分析結果や新たな知見を記録します。
   - `walkthrough.md` 等に結果をまとめ、Gitにコミット・プッシュします。
   - `notify_user` にてユーザーに累積成績のサマリーや特記事項を報告します。
