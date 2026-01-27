import pandas as pd
from modules.scraping import Shutuba, HorseResults, Peds
from modules.preprocessing import DataProcessor, FeatureEngineer

def fetch_and_process_race_data(race_id: str, processor=None, engineer=None, bias_map=None, jockey_stats=None) -> pd.DataFrame:
    """
    指定されたレースIDのデータを取得・加工し、予測用DataFrameを返す
    Args:
        race_id: レースID
        processor: 学習済みのDataProcessor (Noneなら新規作成)
        engineer: 学習済みのFeatureEngineer (Noneなら新規作成)
        bias_map: 学習済みのBias Map (Noneならデフォルト)
        jockey_stats: 学習済みのJockey Stats (Noneならデフォルト)
    """
    print(f"Fetching shutuba data for Race ID: {race_id}...")
    shutuba_df = Shutuba.scrape(race_id)
    
    if shutuba_df.empty:
        raise ValueError(f"Failed to fetch shutuba data for race_id: {race_id}")

    print(f"Found {len(shutuba_df)} horses.")
    
    # 過去成績の取得
    horse_ids = shutuba_df['horse_id'].dropna().unique().tolist()
    print(f"Fetching history for {len(horse_ids)} horses...")
    
    # Check if we have cached horse results (optional, but good for repeatedly running)
    horse_results_df = HorseResults.scrape(horse_ids)
    
    # 血統データの取得
    print(f"Fetching pedigree for {len(horse_ids)} horses...")
    peds_df = Peds.scrape(horse_ids)
    
    if processor is None:
        processor = DataProcessor()
    if engineer is None:
        engineer = FeatureEngineer()
    
    # DataProcessorによる一括処理（race_idからの抽出、数値化、脚質パースなど）
    # process_resultsは内部状態を持たないのでOK
    df = processor.process_results(shutuba_df)
    
    # dateの保証
    if 'date' not in df.columns:
        # race_idから日付を推定する簡易ロジック
        try:
            rid_str = df.index.astype(str)
            year = int(rid_str[0][:4])
            # 月日まではわからないが、適当な日付を入れる（現在日付など）
            # ここでは未来のレースと仮定して、システム日付近い値を入れたいが...
            # preprocessing.pyのロジックだとIndex順で擬似日付を作るが、1レースだけだと意味がない
            # 予測モードでは Interval が計算できればよい。
            # 今回のレース日は、過去成績(Last Date)より新しいことが重要。
            from datetime import datetime
            df['date'] = pd.to_datetime(datetime.now().date())
        except:
            df['date'] = pd.to_datetime('2026-01-01')

    # 基本的な数値変換
    if '単勝' in df.columns: df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(0.0)
    if '人気' in df.columns: df['人気'] = pd.to_numeric(df['人気'], errors='coerce').fillna(0)
    
    # Add History Features & Course Suitability
    if not horse_results_df.empty:
        # Clean horse_results
        horse_results_df = processor.process_results(horse_results_df)
        df = engineer.add_horse_history_features(df, horse_results_df)
        df = engineer.add_course_suitability_features(df, horse_results_df)
    
    # Add Jockey Features
    # 学習済み統計データがあればそれを使用（リーク防止＆精度向上）
    if jockey_stats is not None:
        df, _ = engineer.add_jockey_features(df, jockey_stats)
    else:
        # なければその場のデータでやる（data_loader旧ロジックは削除し、engineerに任せる）
        # ただし engineer.add_jockey_features は jockey_stats=None だと expanding 計算モードになる（が、レース単体だと意味がない）
        # 仕方ないので、historyから簡易計算する（旧ロジック）
        if not horse_results_df.empty and '騎手' in horse_results_df.columns:
             j_results = processor.process_results(horse_results_df)
             j_results = j_results.dropna(subset=['着順'])
             if not j_results.empty:
                 js = j_results.groupby('騎手').agg({
                     '着順': ['mean', lambda x: (x==1).sum() / len(x)]
                 })
                 js.columns = ['jockey_avg_rank', 'jockey_win_rate']
                 if '騎手' in df.columns:
                     df['jockey_avg_rank'] = df['騎手'].map(js['jockey_avg_rank'])
                     df['jockey_win_rate'] = df['騎手'].map(js['jockey_win_rate'])
                     df['jockey_return_avg'] = 75.0 # Default

    # Add Pedigree Features
    if not peds_df.empty:
        df = engineer.add_pedigree_features(df, peds_df)
        
    # Add Bias Features
    # 学習済みマップを使用
    if bias_map is not None:
        df = engineer.add_bias_features(df, bias_map)
    else:
        df['waku_bias_rate'] = 0.3 # Default

    # Add Odds Features (Ensures numeric odds/popularity)
    df = engineer.add_odds_features(df)

    # Encode Categorical
    # processorがlabel_encodersを持っているはず
    categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # encode_categorical は未知のカテゴリを考慮する
    df = processor.encode_categorical(df, categorical_cols)

    # Columns to export matches updated model features + Analysis cols
    export_cols = [
        '枠番', '馬番', '馬名', '斤量', '年齢',
        '体重', '体重変化', 'course_len',
        'avg_rank', 'win_rate', 'place_rate', 'race_count',
        'jockey_avg_rank', 'jockey_win_rate', 'jockey_return_avg',
        'venue_id', 'kai', 'day', 'race_num',
        'avg_last_3f', 'avg_running_style',
        'interval', 'prev_rank',
        'same_distance_win_rate', 'same_type_win_rate',
        'peds_score_speed', 'peds_score_stamina', 'peds_score_dirt',
        'waku_bias_rate',
        'odds', 'popularity',
        'weather', 'ground_state', 'sire', 'dam'
    ]
    
    # Check for missing cols and fill
    for col in export_cols:
        if col not in df.columns:
            if col == 'waku_bias_rate': df[col] = 0.3
            elif col == 'interval': df[col] = 15
            elif col == 'prev_rank': df[col] = 8
            else: df[col] = 0
            
    final_df = df[export_cols].copy()
    # メタデータを継承
    final_df.attrs.update(shutuba_df.attrs)
    return final_df
