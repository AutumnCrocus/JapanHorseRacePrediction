
import os
import pandas as pd
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .constants import RAW_DATA_DIR, RESULTS_FILE, PROCESSED_DATA_DIR
from .scraping import Shutuba, Odds

def load_yearly_data(base_dir, file_prefix, start_year, end_year):
    """
    指定された期間の年別データを読み込み、結合して返す。
    ファイル構成: {base_dir}/{year}/{file_prefix}_{year}.pkl
    """
    years = range(start_year, end_year + 1)
    results = []
    
    def load_single_year(year):
        file_path = os.path.join(base_dir, str(year), f"{file_prefix}_{year}.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # DictならDataFrameへ変換、ListならDataFrameへ
                    if isinstance(data, dict):
                        # 配当データなどの辞書形式の場合
                        return data 
                    elif isinstance(data, list):
                         return pd.DataFrame(data)
                    elif isinstance(data, pd.DataFrame):
                        return data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return None

    # 並列読み込み (I/Oバウンドなので)
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_data = list(executor.map(load_single_year, years))
    
    # 結合処理
    dfs = [d for d in loaded_data if isinstance(d, pd.DataFrame) and not d.empty]
    dicts = [d for d in loaded_data if isinstance(d, dict) and d]
    
    if dfs:
        return pd.concat(dfs)
    elif dicts:
        # 辞書の結合 (キーが重複しない前提、または後勝ち)
        merged_dict = {}
        for d in dicts:
            merged_dict.update(d)
        return merged_dict
    else:
        return pd.DataFrame() # または空のDict? 呼び出し元で判断

def load_payouts(start_year, end_year):
    """配当データを読み込む"""
    payouts_dir = os.path.join(RAW_DATA_DIR, "payouts")
    # 配当データは辞書形式 {race_id: payout_dict}
    data = load_yearly_data(payouts_dir, "payouts", start_year, end_year)
    if isinstance(data, pd.DataFrame) and data.empty:
        return {}
    return data

def load_shutuba(start_year, end_year):
    """出馬表データを読み込む"""
    shutuba_dir = os.path.join(RAW_DATA_DIR, "shutuba")
    return load_yearly_data(shutuba_dir, "shutuba", start_year, end_year)

def load_results(start_year, end_year):
    """
    レース結果データを読み込む。
    年別ファイルがない場合は、既存の results.pickle から該当年をフィルタして返す（移行期間用）。
    """
    results_dir = os.path.join(RAW_DATA_DIR, "results")
    
    # まず年別ファイルを確認
    data = load_yearly_data(results_dir, "results", start_year, end_year)
    
    if (isinstance(data, pd.DataFrame) and not data.empty) or (isinstance(data, dict) and data):
        # Merge DeepFM scores if available
        deepfm_path = os.path.join(PROCESSED_DATA_DIR, 'deepfm_scores.csv')
        if isinstance(data, pd.DataFrame) and os.path.exists(deepfm_path):
             try:
                 print(f"Loading DeepFM scores from {deepfm_path}...")
                 deepfm_df = pd.read_csv(deepfm_path)
                 
                 # Ensure types for merge
                 # Check if race_id exists
                 if 'race_id' in data.columns:
                     data['race_id'] = data['race_id'].astype(str)
                 else:
                     # If race_id is index
                     data = data.reset_index()
                     data['race_id'] = data['race_id'].astype(str)

                 if 'horse_number' in data.columns:
                     data['horse_number'] = data['horse_number'].astype(int)
                 
                 deepfm_df['race_id'] = deepfm_df['race_id'].astype(str)
                 deepfm_df['horse_number'] = deepfm_df['horse_number'].astype(int)
                 
                 # Merge (Left Join)
                 initial_len = len(data)
                 # Keep only necessary columns from deepfm
                 data = pd.merge(data, deepfm_df[['race_id', 'horse_number', 'deepfm_score']], 
                               on=['race_id', 'horse_number'], how='left')
                 print(f"Merged DeepFM scores. Rows: {initial_len} -> {len(data)}")
             except Exception as e:
                 print(f"Failed to merge DeepFM scores: {e}")

        return data
        
    # 年別ファイルがまだない場合、既存の巨大ファイルからロード (フォールバック)
    # ※推奨されないが、移行過渡期のために残す
    print("Warning: Yearly results not found. Loading from monolithic results.pickle (Slow).")
    raw_results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    if os.path.exists(raw_results_path):
        with open(raw_results_path, 'rb') as f:
            full_data = pickle.load(f)
        
        # 年フィルタ
        if isinstance(full_data, pd.DataFrame):
            # dateカラムまたはindexから年を抽出してフィルタ
            # ここでは簡易実装。必要ならデータ構造に合わせて調整
            try:
                if 'date' in full_data.columns:
                    full_data['temp_year'] = pd.to_datetime(full_data['date']).dt.year
                    filtered = full_data[(full_data['temp_year'] >= start_year) & (full_data['temp_year'] <= end_year)]
                    return filtered.drop(columns=['temp_year'])
            except:
                pass
            return full_data # フィルタ失敗したら全量
            
    return pd.DataFrame()

def fetch_and_process_race_data(race_id, processor, engineer, bias_map=None, jockey_stats=None, horse_results_df=None, peds_df=None):
    """
    指定されたレースIDのデータを取得し、前処理・特徴量生成を行ってDataFrameを返す。
    リアルタイム予測(API)用。
    """
    print(f"Fetching data for Race ID: {race_id}")
    
    # 1. 出馬表取得
    df_shutuba = Shutuba.scrape(race_id)
    if df_shutuba.empty:
        print("Failed to fetch shutuba data.")
        return pd.DataFrame()

    # Cleaning: Validate and fix list values in DataFrame
    for col in df_shutuba.columns:
        if df_shutuba[col].apply(lambda x: isinstance(x, list)).any():
            def flatten_cell(x):
                if isinstance(x, list):
                    if len(x) > 0: return str(x[0])
                    else: return ""
                return x
            df_shutuba[col] = df_shutuba[col].apply(flatten_cell)

    # 日付カラムを追加
    if 'date' not in df_shutuba.columns:
        # race_idから日付を推定 (YYYYMMDD...)
        try:
            date_str = str(race_id)[:8]
            df_shutuba['date'] = pd.to_datetime(date_str, format='%Y%m%d')
        except:
            df_shutuba['date'] = datetime.now()

    # 2. オッズ取得
    try:
        odds_data = Odds.scrape(race_id)
        if odds_data and 'tan' in odds_data:
            for idx, row in df_shutuba.iterrows():
                try:
                    umaban = int(row['馬番'])
                    if umaban in odds_data['tan']:
                         df_shutuba.at[idx, '単勝'] = odds_data['tan'][umaban]
                except: pass
    except Exception as e:
        print(f"Warning: Failed to fetch odds: {e}")

    # 3. 前処理 & 特徴量生成
    try:
        # 3.1 Processor
        df_processed = processor.process_results(df_shutuba)
        
        # 3.2 馬体重補完
        if '体重' in df_processed.columns:
            mean_weight = df_processed['体重'].mean()
            if pd.isna(mean_weight): mean_weight = 470.0
            df_processed['体重'] = df_processed['体重'].fillna(mean_weight)
                
        if '体重変化' in df_processed.columns:
            df_processed['体重変化'] = df_processed['体重変化'].fillna(0)

        # 3.3 Engineer
        # 過去成績の統合
        if hasattr(engineer, 'add_horse_history_features') and horse_results_df is not None:
            df_processed = engineer.add_horse_history_features(df_processed, horse_results_df)
            
        if hasattr(engineer, 'add_course_suitability_features') and horse_results_df is not None:
            df_processed = engineer.add_course_suitability_features(df_processed, horse_results_df)
            
        # 騎手成績の統合
        if hasattr(engineer, 'add_jockey_features'):
            if jockey_stats is not None:
                df_processed, _ = engineer.add_jockey_features(df_processed, jockey_stats)
            else:
                df_processed, _ = engineer.add_jockey_features(df_processed)
        
        # 血統データの統合
        if hasattr(engineer, 'add_pedigree_features') and peds_df is not None:
            df_processed = engineer.add_pedigree_features(df_processed, peds_df)
            
        # オッズ・人気特徴量
        if hasattr(engineer, 'add_odds_features'):
            df_processed = engineer.add_odds_features(df_processed)
        
        # トラックバイアス
        if hasattr(engineer, 'add_bias_features') and bias_map is not None:
            df_processed = engineer.add_bias_features(df_processed, bias_map)
        
        # 3.4 カテゴリエンコード
        cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
        cat_cols = [c for c in cat_cols if c in df_processed.columns]
        df_processed = processor.encode_categorical(df_processed, cat_cols)
        
        return df_processed
        
    except Exception as e:
        print(f"Error in processing race data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
