# 開発における教訓とトラブルシューティング (Lessons Learned)

本ドキュメントは、カテゴリごとに以下のファイルへ分割されました。
開発時は各項目に関連するファイルを必ず参照してください。

## カテゴリ別インデックス

1. **[Simulation & Data Leakage](lessons/simulation_data_leakage.md)**
   - シミュレーション時のデータリーク（Data Leak）の罠
   - インデックス形式とカラム欠落
   - データリーク抜本対策（過去データと推論データの統合）

2. **[Pandas & Data Processing](lessons/pandas_data_processing.md)**
   - Pandas `groupby` の挙動
   - データ型 (`dtypes`) の管理
   - カラム名の空白・タイポ
   - `read_html` のテーブル選択
   - データローディングと辞書構造

3. **[Web Scraping & Automation](lessons/web_scraping_automation.md)**
   - SPA(Riot.js)の待機処理
   - 流し投票のDOM構造
   - 金額入力（単価vs合計）
   - 馬番のゼロ埋め
   - HTML改行文字の処理

4. **[Strategies & Models](lessons/strategies_models.md)**
   - 3連単BOXの計算ロジック
   - フォーメーション変換
   - `_format_recommendations` のフィールド欠落
   - BOX頭数と予算の線形性
   - LTRモデルのスコア正規化 (Softmax)

5. **[Development & CI](lessons/development_ci.md)**
   - ログ出力
   - CI/Lintエラー対応
   - フロントエンド検証環境
   - Flake8対応
