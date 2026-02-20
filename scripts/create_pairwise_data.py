
import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_dataset():
    """データセットのロード"""
    file_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_2010_2025.pkl')
    if not os.path.exists(file_path):
         file_path = os.path.join(RAW_DATA_DIR, 'processed', 'dataset_2010_2025.pkl')
    
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
        
    df = data_dict['data']
    # 日付変換
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    # 日付変換
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    # race_idがない場合は作成 (original_race_idから)
    if 'race_id' not in df.columns and 'original_race_id' in df.columns:
        df['race_id'] = df['original_race_id'].astype(str)
    
    # Filter for training period (e.g. 2021-2024)
    # Exclude 2025 to prevent data leakage during simulation!
    print("Filtering dataset for training (2021-2024)...")
    df = df[(df['date'].dt.year >= 2021) & (df['date'].dt.year <= 2024)].copy()
    print(f"Dataset size after filtering: {len(df)}")
        
    return df

def create_pairwise_data(df, output_dir):
    """
    レース結果をペアワイズデータに変換する
    choix入力形式: LIST OF (winner, loser) tuples
    """
    
    print("Pre-processing data for Bayesian model...")
    
    # 馬IDのエンコーディング (0 ~ N-1)
    # 名前ではなくhorse_idを使うべきだが、欠損がある場合は名前でフォールバック
    if 'horse_id' not in df.columns:
        print("Warning: horse_id not found, using horse_name as ID.")
        df['horse_id'] = df['horse_name']
        
    unique_horses = df['horse_id'].unique()
    n_horses = len(unique_horses)
    print(f"Total unique horses: {n_horses}")
    
    # IDマップ作成
    horse_to_id = {h: i for i, h in enumerate(unique_horses)}
    id_to_horse = {i: h for h, i in horse_to_id.items()}
    
    # データをレースごとにグループ化
    # 2010-2024を学習用、2025をテスト用とするが、Rating学習には全データを使うのがベイズ的
    # ただし、時系列を考慮して「過去のレースのみ」から学習するべきか？
    # Plackett-Luceの静的モデルでは全期間の強さを1つの値で表すため、ここでは全期間を使用する。
    # (動的モデルにするならGlickoなどが必要だが、今回はchoixの静的モデルを採用)
    
    # 着順データのクリーニング
    # '着順' カラムが必要
    if '着順' not in df.columns:
        raise ValueError("Dataset missing '着順' column")
        
    # 着順を数値に変換 (除外、中止などはNaNまたは大きな値に)
    def clean_rank(x):
        try:
            return int(x)
        except:
            return 999
            
    df['rank_clean'] = df['着順'].apply(clean_rank)
    
    # 有効な着順のみ抽出 (1-18着くらい)
    valid_df = df[df['rank_clean'] <= 20].copy()
    
    comparisons = []
    
    grouped = valid_df.groupby('race_id')
    print(f"Processing {len(grouped)} races...")
    
    for rid, group in tqdm(grouped):
        # 着順でソート
        sorted_group = group.sort_values('rank_clean')
        
        # 同着がある場合の処理は複雑だが、choixは単純なペアワイズかランキングリスト
        # ここでは単純化して、上位の馬は下位の馬すべてに勝ったとする (Plackett-Luceの分解)
        # あるいは、隣接ペアのみを入れるか？
        # choixドキュメント推奨: 1位は2位に勝ち、2位は3位に勝ち... というチェインではなく
        # 1位は2位,3位...全てに勝つ、という全ペア生成が一般的だが計算量が増える。
        # RANKING LIST形式 (choix.ilsr_pairwise ではなく ilsr_rankings を使う手もある)
        
        # ここでは「ランキングリスト」を作成し、あとでchoixに任せる
        # horse_ids in rank order
        ranking = []
        for _, row in sorted_group.iterrows():
            h_id = row['horse_id']
            if h_id in horse_to_id:
                ranking.append(horse_to_id[h_id])
                
        if len(ranking) >= 2:
            comparisons.append(ranking)
            
    print(f"Generated {len(comparisons)} rankings.")
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'rankings': comparisons,
        'horse_to_id': horse_to_id,
        'id_to_horse': id_to_horse,
        'n_horses': n_horses
    }
    
    out_path = os.path.join(output_dir, 'bayesian_data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved Bayesian data to {out_path}")
    return out_path

if __name__ == "__main__":
    df = load_dataset()
    
    # モデルディレクトリの下に保存
    output_dir = os.path.join(os.path.dirname(PROCESSED_DATA_DIR), 'models', 'bayesian')
    create_pairwise_data(df, output_dir)
