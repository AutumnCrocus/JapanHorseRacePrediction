
import os
import sys
import pickle
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.constants import MODEL_DIR

def main():
    MODEL_PATH = os.path.join(MODEL_DIR, "standalone_ranking", "ranking_model.pkl")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found at", MODEL_PATH)
        return

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # 最新のテストデータを読み込み（検証用）
    # 簡易的に、学習スクリプトで作成される中間データがあればそれを使うが、
    # ここでは既存の results.pickle から1レース抽出して試す
    from scripts.training.train_ranking_standalone import prepare_ranking_labels
    from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
    from modules.preprocessing import DataProcessor, FeatureEngineer

    print("検証用のデータをロード中...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, PEDS_FILE), 'rb') as f:
        peds = pickle.load(f)

    # データ整形 (train_ranking_standalone.py と同様のロジック)
    if isinstance(results.index, pd.Index) and results.index.name == 'race_id':
        results = results.reset_index()
    elif 'race_id' not in results.columns:
        results['race_id'] = results.index.astype(str)
        
    # カラム重複除去
    results = results.loc[:, ~results.columns.duplicated()]

    if 'date' not in results.columns:
        results['date'] = pd.to_datetime(results['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
    results['date'] = pd.to_datetime(results['date']).dt.normalize()
    
    if '--date' in sys.argv:
        date_str = sys.argv[sys.argv.index('--date') + 1]
        test_date = pd.to_datetime(date_str).normalize()
        race_ids = results[results['date'] == test_date]['race_id'].unique()
        if len(race_ids) == 0:
            print(f"No races found on {date_str}. Using latest instead.")
            test_race_id = results['race_id'].unique()[-1]
        else:
            test_race_id = race_ids[0]
    else:
        # 直近の2025年の1レースを抽出
        races_2025 = results[results['date'].dt.year == 2025]['race_id'].unique()
        if len(races_2025) > 0:
            test_race_id = races_2025[-1]
        else:
            test_race_id = results['race_id'].unique()[-1]
    
    race_df = results[results['race_id'] == test_race_id].copy()
    
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 結果の表示用データを別途保持
    display_df = race_df.copy()
    if 'race_id' not in display_df.columns: display_df = display_df.reset_index()
    
    # 特徴量作成
    df_proc = processor.process_results(race_df)
    df_proc = engineer.add_horse_history_features(df_proc, hr)
    df_proc = engineer.add_course_suitability_features(df_proc, hr)
    df_proc, _ = engineer.add_jockey_features(df_proc)
    df_proc = engineer.add_pedigree_features(df_proc, peds)
    df_proc = engineer.add_odds_features(df_proc)
    
    # 予測に必要な特徴量を準備
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    X = df_proc[feature_names]
    scores = model.predict(X)
    
    # 結果の統合
    display_df = display_df.iloc[:len(scores)].copy() # 行数を合わせる
    display_df['ranking_score'] = scores
    display_df['actual_rank'] = display_df['rank'] if 'rank' in display_df.columns else (display_df['着順'] if '着順' in display_df.columns else 0)
    
    # 表示カラムの整理
    cols = ['horse_name', 'ranking_score', 'actual_rank']
    if 'umaban' in display_df.columns: cols.insert(0, 'umaban')
    elif '馬番' in display_df.columns: cols.insert(0, '馬番')
    
    # 存在するカラムのみを使用
    final_cols = [c for c in cols if c in display_df.columns]
    
    result = display_df[final_cols].sort_values('ranking_score', ascending=False)
    print("\n--- 予測結果 (スコア順) ---")
    print(result.to_string(index=False))
    
    # NDCG簡易計算 (1位的中チェック)
    top_pred_horse = result.iloc[0]
    if top_pred_horse['actual_rank'] == 1:
        print("\n✅ 見事！予測1位が実際の1位でした。")
    else:
        print(f"\n❌ 予測1位の実際の着順: {top_pred_horse['actual_rank']}着")

if __name__ == "__main__":
    main()
