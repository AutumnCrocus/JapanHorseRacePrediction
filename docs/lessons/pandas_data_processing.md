# Pandas & Data Processing Lessons

## 1. Pandas `groupby` の挙動に関する誤解 (2026-02-02)

* **事象**: `df.groupby(level=0)` を使用した際、インデックスがユニークなID（`race_id`）だと思い込んでいたが、実際には`race_id`がインデックスに設定されていなかった。
* **教訓**: `groupby` を使用する際は、インデックス(`level=...`)に頼らず、**明示的にカラム名(`by='race_id'`)を指定する**方が安全で可読性も高い。
* **対策**: `df.groupby('race_id')` のようにカラム名を指定する。

## 2. Pandas データ型 (`dtypes`) の厳密な管理 (2026-02-02)

* **事象**: `model.predict` にデータを渡す際、一部のカラム（`枠番`, `馬番`など）が `object` 型のまま渡され、モデルがエラーを吐いた。
* **教訓**: 予測直前（Just-In-Time）に、特徴量カラムに対して**強制的に数値変換を行う処理**を挟むのが最も確実である。
* **対策**:

    ```python
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    preds = model.predict(X)
    ```

## 3. カラム名のタイポと空白文字 (2026-02-02)

* **事象**: `race_data['馬 番']` というように、カラム名に余計なスペースが含まれているコードが存在し、`KeyError` を引き起こした。
* **教訓**: 日本語カラム名は視認しづらいため、`df.columns` をプリントして確認する。エラーハンドリング (`try-except`) は、**開発中は具体的にエラー内容を標準出力/ログに出す**。

## 4. `pd.read_html` の仕様とテーブル選択の罠 (2026-02-14)

* **事象**: `pd.read_html` はページ内の全テーブルをリストで返すが、コードが特定のキーワードを含む最初のテーブルだけを返して終了していたため、単勝や馬連の情報が含まれる別のテーブルが無視されていた。
* **教訓**: `pd.read_html` の結果をループ処理する際は、**「条件に合致する全てのテーブル」を収集して結合 (`pd.concat`) する**設計にする。
* **対策**: 取得した全てのデータフレームに対して、払い戻しに関連するキーワードが含まれているかを検査し、該当するものを全てリストに格納した上で結合する。

## 5. pandas `FutureWarning` と実行時ノイズ (2026-02-13)

* **事象**: `fillna` や `astype` に伴う `FutureWarning` が大量に出力され、本来のエラーや進捗が見づらくなっていた。
* **教訓**: 本番運用やレポート生成スクリプトでは、**既知の安全な警告は積極的に抑制 (warnings.filterwarnings)** し、実行ログの視認性を高く保つ。
* **対策**: `warnings.simplefilter(action='ignore', category=FutureWarning)` や `pd.set_option('future.no_silent_downcasting', True)` を設定する。

## 6. シミュレーション時のデータローディングと辞書構造 (2026-02-19)

* **事象**: `simulate_ltr_strategies.py` において、`calc_return` 関数が `TypeError` でクラッシュした。
* **原因**: 払い戻しデータ（`returns_map`）の構造が、単勝/複勝では `{horse_number: payout}` の辞書、馬連/ワイド等では `{(h1, h2): payout}` の辞書（タプルキー）であったが、シミュレーションロジックの一部が `list of dicts` 形式（旧仕様？）を想定してループ処理を書いていたため。
* **教訓**: データの読み込み元（`load_payouts` vs `pickle.load`）によってデータ構造が微妙に異なる場合がある。
* **対策**: `winning_data` が辞書型かリスト型かを判定するか、統一されたデータローダー（`modules.data_loader`）を使用し、その戻り値の仕様を正しく実装に反映させる。今回は辞書型のアイテム反復 (`.items()`) に修正して解決した。
