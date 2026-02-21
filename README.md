# 競馬予想AI

機械学習を活用した競馬予想システム。LightGBMモデルによる高精度な予測を提供します。

## 機能

- **データ収集**: netkeiba.comからレース結果、馬の成績、血統、払戻データをスクレイピング
- **機械学習予測**: LightGBM（lgbm）/ Learning-to-Rank（ltr）/ スタッキングアンサンブル（stacking）の3モデルを搭載
- **自動投票**: JRA即PATへの自動投票機能（OrePro経由）
- **回収率シミュレーション**: 複数モデル×複数戦略のクロスシミュレーション
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

<http://localhost:5000> にアクセスしてWeb UIを使用。
IPAT連携機能を使用する場合は、予測結果画面から「IPATで馬券を購入」ボタンをクリックし、即PATの認証情報を入力してください。
※認証情報はブラウザ内に一時保存可能ですが、セキュリティ上の理由から保存しないことを推奨します。

## 戦略別パフォーマンス (2025年 全期間シミュレーション)

2025年全レース（3,455R）を対象に、3モデル×複数戦略のシミュレーションを実施した結果です。
予算は5,000円/レース固定。

### モデル × 戦略 ランキング（回収率順）

| 順位 | モデル | 戦略 | 回収率 | 収益 | 的中率 |
|---:|:---|:---|---:|---:|---:|
| **1** | **lgbm** | **box4_sanrenpuku** | **211.5%** | +1,540,240円 | 17.7% |
| 2 | ltr | box4_sanrenpuku | 207.1% | +1,480,760円 | 16.3% |
| 3 | stacking | box4_sanrenpuku | 199.6% | +1,376,730円 | 28.1% |
| 4 | stacking | sanrenpuku_1axis | 153.4% | +2,855,780円 | 52.6% |
| 5 | lgbm | sanrenpuku_1axis | 145.6% | +2,435,840円 | 35.2% |
| 6 | ltr | sanrenpuku_1axis | 144.9% | +2,396,580円 | 33.3% |
| 7 | ltr | wide_nagashi | 133.1% | +570,620円 | 53.2% |
| 8 | ltr | ranking_anchor | 127.7% | +4,791,790円 | 52.7% |
| 9 | stacking | ranking_anchor | 126.3% | +4,549,404円 | 66.9% |
| 10 | lgbm | wide_nagashi | 125.7% | +444,380円 | 50.9% |

> **★推奨**: `lgbm` × `box4_sanrenpuku`（回収率211.5%）
> ※ `formation` 戦略は全モデルで回収率30〜33%と不振のため非推奨。

### 選択可能な戦略一覧

| 戦略名 | 券種 | 買い方（点数） | 特徴 |
|:---|:---|:---|:---|
| **box4_sanrenpuku** | 3連複 | 上位4頭BOX（4点） | ★最推奨。コスト400円で高配当を狙う |
| box5_sanrenpuku | 3連複 | 上位5頭BOX（10点） | 的中率重視。1点あたり100円 |
| sanrenpuku_1axis | 3連複 | 軸1頭流し（15点） | 高的中率。収益総額が大きい |
| sanrenpuku_2axis | 3連複 | 軸2頭流し | 軸の組み合わせで絞り込む |
| box4_umaren | 馬連 | 上位4頭BOX（6点） | 低コスト・安定型 |
| umaren_nagashi | 馬連 | 軸1頭流し（5点） | 軸馬の信頼度が高い場合向け |
| wide_nagashi | ワイド | 軸1頭流し（5点） | 的中率最高。回収率は控えめ |
| ranking_anchor | 3連複 | アンカー馬軸流し | 大量投票向け。LTRモデルと相性良 |
| hybrid_1000 | 3連複/ワイド | 混戦判定で切り替え | 予算1,000円専用 |
| kelly | 単勝/馬連/3連複 | 期待値最大化 | オッズデータ必要 |
| odds_divergence | 単勝中心 | AI vs 市場乖離狙い | オッズデータ必要 |
| formation | 3連単 | 上位3頭BOX（6点） | **現在非推奨**（回収率30%台） |
