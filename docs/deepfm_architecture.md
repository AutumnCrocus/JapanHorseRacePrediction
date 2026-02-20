# DeepFM (Deep Factorization Machines) Architecture for Horse Racing

## 概要

**DeepFM** は、**Factorization Machines (FM)** と **Deep Neural Networks (DNN)** を統合したニューラルネットワークモデルです。
主にCTR（Click-Through Rate）予測などのレコメンデーション分野で使われますが、競馬予測においては「**Cold Start問題（新規馬・騎手の予測困難性）**」や「**疎な特徴量（IDなど）の相互作用**」を解決するために導入しました。

## なぜDeepFMなのか？

従来のGBDT（LightGBMなど）は強力ですが、以下の弱点があります：

1. **高次元の疎なデータ（Sparse Data）に弱い**: 何千もの「馬ID」「騎手ID」をそのまま扱うのが難しい（One-hot Encodingすると次元爆発する）。
2. **未知の組み合わせ（Cold Start）**: 「この騎手がこの馬に乗るのは初めて」といった場合、学習データにその組み合わせがないと予測が難しい。

DeepFMはこれらを解決します：

- **埋め込み層（Embedding Layer）**: IDを低次元のベクトル（例: 64次元）に変換し、似た特徴を持つID同士を近くに配置します。
- **相互作用の学習**: FM成分が「馬×騎手」「種牡馬×コース」といった2次の相互作用を、DNN成分が高次の非線形相互作用を学習します。

## アーキテクチャ詳細

DeepFMは、同じ入力（Embedding）を共有する2つのコンポーネントから成ります。

### 1. FM Component (Factorization Machine)

- **役割**: 低次（2次）の特徴量相互作用をモデル化します。
- **仕組み**: 特徴量 $i$ と $j$ の相互作用を、それぞれの潜在ベクトル $V_i, V_j$ の内積 $\langle V_i, V_j \rangle x_i x_j$ で表現します。
- **効果**: 「この騎手は逃げ馬と相性が良い」「この種牡馬は重馬場に強い」といった関係性を、具体的な組み合わせデータが少なくても、潜在ベクトルの類似性から推論できます。

### 2. Deep Component (DNN)

- **役割**: 高次の特徴量相互作用をモデル化します。
- **仕組み**: Embedding層の出力を連結し、多層パーセプトロン（MLP）に入力します。
- **効果**: 複雑で非線形なパターン（例: 「距離短縮」かつ「外枠」かつ「騎手乗り替わり」のときの勝率変化など）を学習します。

### 3. Output

$$
y_{DeepFM} = \text{sigmoid}(y_{FM} + y_{DNN})
$$
両方の出力を合算し、シグモイド関数で確率（0~1）に変換します。

## このプロジェクトでの実装 (`modules/models/deepfm.py`)

- **Sparse Features**: `race_id`, `horse_id`, `jockey_id`, `trainer_id`, `owner_id` 等のID群。これらはEmbedding層で学習されます。
- **Dense Features**: `jockey_win_rate`, `horse_recent_time` 等の数値データ。これらも同じ空間に射影され、相互作用に参加します。
- **Cold Start対策**: 新規の馬や騎手が来ても、「未知のID」として扱うのではなく、その他の属性（血統、所属厩舎など）のEmbeddingを通じて予測が可能になります（※完全な新規IDの場合はUnknownトークンを使用）。

## まとめ

DeepFMは、**「IDごとの個性を捉えつつ（FM）、複雑な条件の組み合わせも学習する（DNN）」** ハイブリッドなモデルです。特にデータが疎（Sparse）になりがちな競馬データの特性にマッチしています。
