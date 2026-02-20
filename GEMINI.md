# ユーザー定義ルール

以下のルールを遵守すること。

1. **日本語での応答**
   - ユーザーへの応答はすべて日本語で行うこと。
   - `notify_user` のメッセージや、Artifact内の記述も日本語とする。

2. **日本語ドキュメント**
   - 実装計画 (`implementation_plan.md`) やタスクリスト (`task.md`) など、ユーザーが確認するファイルは日本語で記述すること。

3. **日本語文への記号混入防止**
   - 日本語の文章中に `\n` などの改行コードや制御文字がそのまま表示されないようにすること。
   - 適切な改行や段落分けを行うこと。
4. **並列処理の禁止**
   - スクレイピングやデータ処理において、`ProcessPoolExecutor` や `ThreadPoolExecutor` などの並列実行は原則禁止とする。
   - PC負荷増大を防ぎ、サーバーへの過度なアクセスを避けるため、逐次処理 (Sequential Processing) を基本とする。

5. **GitHub Actionsの確認**
   - コミット後、必ずGitHub Actions ( <https://github.com/AutumnCrocus/JapanHorseRacePrediction/actions> ) を確認し、エラーが発生していないかチェックすること。
6. **過去の教訓 (Lessons Learned) の参照と蓄積**
   - 実装や設計を行う前に、必ず `docs/lessons/` 配下の関連する教訓ファイルを確認すること。
   - **教訓の蓄積**: 改修に時間を要した不具合や、新たな知見が得られた場合は、必ず `docs/lessons/` 配下の適切なカテゴリのファイルにその内容（事象・原因・対策）を追記すること。既存のカテゴリに当てはまらない場合は新規ファイルを作成すること。
   - 特に以下のファイルは頻繁に参照すること。
     - `docs/lessons/pandas_data_processing.md`: データ処理のハマりどころ
     - `docs/lessons/simulation_data_leakage.md`: データリーク防止
     - `docs/lessons/web_scraping_automation.md`: 自動化の定石
