# 競馬予想AI

機械学習を活用した競馬予想システム。LightGBMモデルによる高精度な予測を提供します。

## 機能

- **データ収集**: netkeiba.comからレース結果、馬の成績、血統、払戻データをスクレイピング
- **機械学習予測**: LightGBMによる複勝確率予測
- **IPAT連携**: JRA即PATへの自動投票機能（実験的機能）
- **回収率シミュレーション**: 様々な賭け戦略のシミュレーション
- **Web UI**: レース予測結果をわかりやすく表示

## ディレクトリ構成

```
ネット競馬/
├── main.py              # CLIメインスクリプト
├── app.py               # Flask Webアプリケーション
├── requirements.txt     # Python依存パッケージ
├── modules/
│   ├── __init__.py
│   ├── constants.py     # 定数定義
│   ├── ipat_connector.py # IPAT連携モジュール
│   ├── scraping.py      # スクレイピングモジュール
│   ├── preprocessing.py # データ前処理
│   ├── training.py      # 機械学習モデル
│   └── simulation.py    # 回収率シミュレーション
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── data/                # データ保存ディレクトリ（自動生成）
│   ├── raw/
│   └── processed/
└── models/              # モデル保存ディレクトリ（自動生成）
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. デモの実行（推奨）

```bash
python main.py demo
```

サンプルデータでモデルを作成し、予測結果を表示します。

## 使い方

### CLIモード

```bash
# データスクレイピング（時間がかかります）
python main.py scrape --year 2024 --sample

# モデル学習
python main.py train

# 予測実行
python main.py predict
```

### Webアプリモード

```bash
python app.py
```

http://localhost:5000 にアクセスしてWeb UIを使用。
IPAT連携機能を使用する場合は、予測結果画面から「IPATで馬券を購入」ボタンをクリックし、即PATの認証情報を入力してください。
※認証情報はブラウザ内に一時保存可能ですが、セキュリティ上の理由から保存しないことを推奨します。

## モデルについて

### アルゴリズム
- **LightGBM**: 勾配ブースティング決定木を使用
- **目的変数**: 複勝（3着以内かどうかの二値分類）

### 主な特徴量
- 人気、単勝オッズ、斤量
- 馬の過去成績（平均着順、勝率、複勝率、出走回数）
- 騎手成績（平均着順、勝率）
- 性別、年齢、馬体重
- コース距離、コース種別

## 注意事項

⚠️ **免責事項**
- 本システムは予測を保証するものではありません
- 馬券購入は自己責任でお願いします
- netkeiba.comへのスクレイピングは適切な間隔を空けて行ってください
- **IPAT連携機能について**: 本機能は補助ツールであり、正常な投票を保証しません。必ず投票履歴をJRA公式サイトで確認してください。

## 参考

- [競馬予想で始める機械学習〜完全版〜](https://zenn.dev/dijzpeb/books/848d4d8e47001193f3fb)
