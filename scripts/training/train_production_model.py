import os
import sys
import pickle
import pandas as pd
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import (
    RAW_DATA_DIR, MODEL_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
)
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def train_production_model():
    print("=== 本番モデル再学習開始 ===")
    
    # 1. データの読み込み
    print("\n[1/4] データを読み込み中...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    horse_results_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results = pickle.load(open(results_path, 'rb'))
    horse_results = pickle.load(open(horse_results_path, 'rb'))
    peds = pickle.load(open(peds_path, 'rb'))
    
    print(f"  - レース結果: {len(results)} 行")
    print(f"  - 馬の成績: {len(horse_results)} 行")
    print(f"  - 血統データ: {len(peds)} 件")
    
    # 2. 前処理と特徴量エンジニアリング
    print("\n[2/4] 前処理と特徴量エンジニアリングを実行中...")
    # prepare_training_data は (X, y, processor, engineer, bias_map, jockey_stats, df) を返す
    X, y, processor, engineer, bias_map, jockey_stats, _ = prepare_training_data(
        results, horse_results, peds, scale=False
    )
    
    # original_race_id は文字列なので学習から除外
    if 'original_race_id' in X.columns:
        X = X.drop(columns=['original_race_id'])
    
    print(f"  - 特徴量数: {len(X.columns)}")
    print(f"  - 学習サンプル数: {len(X)}")
    
    # 3. モデルの学習
    print("\n[3/4] モデルを学習中 (LightGBM)...")
    model = HorseRaceModel(model_type='lgbm')
    # 本格的な学習のため test_size=0.1 (10%を検証用に使用して過学習を抑制)
    metrics = model.train(X, y, test_size=0.1)
    
    print("\n学習完了。評価指標 (検証データ):")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\n特徴量重要度 (上位10):")
    importance = model.get_feature_importance(10)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
        
    # 4. アーティファクトの保存
    print("\n[4/4] アーティファクトを保存中...")
    
    # 既存ファイルのバックアップ（念のため）
    files_to_save = {
        'production_model.pkl': model,
        'processor.pkl': processor,
        'engineer.pkl': engineer,
        'bias_map.pkl': bias_map,
        'jockey_stats.pkl': jockey_stats
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for filename, obj in files_to_save.items():
        target_path = os.path.join(MODEL_DIR, filename)
        
        # バックアップ
        if os.path.exists(target_path):
            bak_path = target_path + ".bak_" + timestamp
            os.rename(target_path, bak_path)
            # print(f"  - バックアップ作成: {filename} -> {os.path.basename(bak_path)}")
            
        # 保存
        if hasattr(obj, 'save') and callable(getattr(obj, 'save')):
            # HorseRaceModel など save メソッドを持つ場合
            obj.save(target_path)
        else:
            # それ以外は pickle
            with open(target_path, 'wb') as f:
                pickle.dump(obj, f)
        
        print(f"  - 保存完了: {filename}")
        
    print("\n=== 本番モデル再学習プロセス完了 ===")

if __name__ == "__main__":
    train_production_model()
