"""
データ前処理モジュール
スクレイピングデータの加工・特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .constants import (
    RACE_TYPE_DICT, WEATHER_DICT, GROUND_STATE_DICT,
    SEX_DICT, RACE_TYPE_MAP, WEATHER_MAP, GROUND_MAP
)


class DataProcessor:
    """データ前処理クラス"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.scaled_columns = []
    
    def process_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        レース結果データを前処理
        
        Args:
            results_df: 生のレース結果DataFrame
            
        Returns:
            前処理済みのDataFrame
        """
        df = results_df.copy()
        
        # カラム名の正規化
        df.columns = df.columns.str.replace(' ', '')
        
        # 着順を数値化（除外・中止・取消などは欠損値に）
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            
            # 着順に欠損がある行を削除
            df = df.dropna(subset=['着順'])
            df['着順'] = df['着順'].astype(int)
        
        # 枠番・馬番を数値化
        if '枠番' in df.columns:
            df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(0).astype(int)
        if '馬番' in df.columns:
            df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(0).astype(int)

        if 'course_len' not in df.columns:
            df['course_len'] = 2000 # Default distance
        if 'race_type' not in df.columns:
            df['race_type'] = 0 # Default (Turf)
            
        # race_id (Index) から情報を抽出 (YYYY PP KK DD RR)
        try:
            # Preserve Index as Column for later restoration
            df['original_race_id'] = df.index.astype(str)
            race_ids = df.index.astype(str)
            df['venue_id'] = race_ids.str[4:6].astype(int)
            df['kai'] = race_ids.str[6:8].astype(int)
            df['day'] = race_ids.str[8:10].astype(int)
            df['race_num'] = race_ids.str[10:12].astype(int)
        except Exception:
            df['venue_id'] = 0
            df['kai'] = 0
            df['day'] = 0
            df['race_num'] = 0

        # 性齢を分離して数値化
        if '性齢' in df.columns:
            df['性'] = df['性齢'].str[0].map(SEX_DICT).fillna(0).astype(int)
            df['年齢'] = pd.to_numeric(df['性齢'].str[1:], errors='coerce').fillna(4).astype(int)
        elif '性' in df.columns:
            # すでに性カラムがある場合 (API経由など)
            df['性'] = df['性'].map(lambda x: SEX_DICT.get(x, x) if isinstance(x, str) else x).fillna(0).astype(int)
        
        # カテゴリ項目のマッピング (文字列の場合に数値へ変換)
        if 'race_type' in df.columns:
            df['race_type'] = df['race_type'].map(lambda x: RACE_TYPE_MAP.get(x, x) if isinstance(x, str) else x).fillna(0).astype(int)
        if 'weather' in df.columns:
            df['weather'] = df['weather'].map(lambda x: WEATHER_MAP.get(x, x) if isinstance(x, str) else x).fillna(0).astype(int)
        if 'ground_state' in df.columns:
            df['ground_state'] = df['ground_state'].map(lambda x: GROUND_MAP.get(x, x) if isinstance(x, str) else x).fillna(0).astype(int)

        # 斤量を数値化
        if '斤量' in df.columns:
            df['斤量'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(56.0)
        
        # タイムを秒に変換
        if 'タイム' in df.columns:
            df['タイム秒'] = df['タイム'].apply(self._time_to_seconds)
        
        # 体重と体重変化を分離
        if '馬体重' in df.columns:
            df['体重'] = df['馬体重'].apply(self._extract_weight)
            df['体重変化'] = df['馬体重'].apply(self._extract_weight_change)
        
        # 単勝オッズを数値化
        if '単勝' in df.columns:
            df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce')
        
        # 人気を数値化
        if '人気' in df.columns:
            df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
        
        # 上り3Fを数値化
        if '上り' in df.columns:
            # カラム名を統一（半角スペースなどを考慮）
            df['上り'] = pd.to_numeric(df['上り'], errors='coerce')
        
        # 通過順位を数値化（脚質指標）
        if '通過' in df.columns:
            df['running_style'] = df['通過'].apply(self._parse_running_style)
            # 头数（レースの出走頭数）があれば正規化する
            if '頭数' in df.columns:
                df['頭数'] = pd.to_numeric(df['頭数'], errors='coerce')
                df['running_style'] = df['running_style'] / df['頭数']

        return df
    
    def _parse_running_style(self, passage_str) -> float:
        """通過順位文字列から脚質（位置取り）を数値化 (例: 1-1-2-2 -> 1.5)"""
        if pd.isna(passage_str) or not isinstance(passage_str, str):
            return np.nan
        try:
            # "4-4-3-2" のような形式をパース
            positions = re.findall(r'\d+', passage_str)
            if not positions:
                return np.nan
            # 数値に変換して平均を取る（全てのコーナーでの平均位置）
            pos_ints = [int(p) for p in positions]
            return sum(pos_ints) / len(pos_ints)
        except:
            return np.nan
    
    def _time_to_seconds(self, time_str) -> float:
        """タイム文字列を秒に変換"""
        if pd.isna(time_str):
            return np.nan
        try:
            time_str = str(time_str)
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except:
            return np.nan
    
    def _extract_weight(self, weight_str) -> float:
        """体重を抽出"""
        if pd.isna(weight_str):
            return np.nan
        try:
            match = re.match(r'(\d+)', str(weight_str))
            if match:
                return float(match.group(1))
            return np.nan
        except:
            return np.nan
    
    def _extract_weight_change(self, weight_str) -> float:
        """体重変化を抽出"""
        if pd.isna(weight_str):
            return np.nan
        try:
            match = re.search(r'\(([+-]?\d+)\)', str(weight_str))
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return np.nan
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        カテゴリ変数をラベルエンコード
        
        Args:
            df: DataFrame
            columns: エンコードするカラムのリスト
            
        Returns:
            エンコード済みのDataFrame
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                # 欠損値を'unknown'で埋める
                df[col] = df[col].fillna('unknown').astype(str)
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # 未知のカテゴリは-1として処理
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in known_classes else -1
                    )
        
        return df

    def fit_scale(self, df: pd.DataFrame):
        """StandardScalerをfit"""
        self.scaler = StandardScaler()
        self.scaled_columns = df.columns.tolist()
        self.scaler.fit(df)
        
    def transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """スケーリング適用"""
        if self.scaler is None:
            return df
            
        df_scaled = df.copy()
        # カラムが一致することを確認
        cols = [c for c in self.scaled_columns if c in df_scaled.columns]
        if cols:
             # transform returns numpy array, putting back to dataframe
             df_scaled[cols] = self.scaler.transform(df_scaled[cols])
             
        return df_scaled


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""
    
    def __init__(self):
        pass
    
    def _preprocess_horse_results(self, horse_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        horse_results_dfに日付情報を追加し、時系列順にソートする
        
        Args:
            horse_results_df: 馬の過去成績DataFrame
            
        Returns:
            日付カラム追加・ソート済みのDataFrame
        """
        df = horse_results_df.copy()
        
        # レース名から日付を抽出（YY/MM/DD形式）
        df['date_str'] = df['レース名'].str.extract(r'(\d{2}/\d{2}/\d{2})')[0]
        
        # YY/MM/DD -> YYYY-MM-DD に変換（2000年以降と仮定）
        df['date'] = pd.to_datetime('20' + df['date_str'], format='%Y/%m/%d', errors='coerce')
        
        # 日付順にソート
        df = df.sort_values(['date'], ascending=True)
        
        return df
    
    def add_horse_history_features(self, df: pd.DataFrame, horse_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        馬の過去成績から特徴量を追加（データリーク修正版 - ローリング平均+シフト+厳密マージ）
        """
        df = df.copy()
        
        # horse_id が index にある場合、カラムに持ってくる (インデックス名が None の場合も考慮)
        if 'horse_id' not in df.columns:
            if df.index.name == 'horse_id' or 'horse_id' in df.index.names:
                df = df.reset_index(level='horse_id')
            elif df.index.name is None and not isinstance(df.index, pd.MultiIndex):
                # 無名の1次インデックスを horse_id とみなす (慣習的な実装)
                df = df.reset_index().rename(columns={'index': 'horse_id'})
        
        # デフォルト値
        default_values = {
            'avg_rank': 8.0, 'win_rate': 0.08, 'place_rate': 0.2,
            'race_count': 0, 'avg_last_3f': 37.0, 'avg_running_style': 0.5,
            'interval': 20, 'prev_rank': 8.0
        }
        for col, val in default_values.items():
            if col not in df.columns:
                df[col] = val
        
        # date カラムが欠落している場合の補助 (特に推論用データ)
        if 'date' not in df.columns:
            # original_race_id もしくは index から取得
            rid_source = None
            if 'original_race_id' in df.columns: rid_source = df['original_race_id']
            elif df.index.name == 'race_id' or (not df.index.name and not isinstance(df.index, pd.MultiIndex)):
                rid_source = df.index.to_series()
            
            if rid_source is not None:
                # YYYYMMDD または YYYYPP... 形式を想定して先頭8文字を日付とする
                df['date'] = pd.to_datetime(rid_source.astype(str).str[:8], format='%Y%m%d', errors='coerce').dt.normalize()
            else:
                # 最終手段: 現在日付
                df['date'] = pd.Timestamp.now().normalize()

        if horse_results_df is None or horse_results_df.empty or 'horse_id' not in df.columns:
            return df
        
        # 1. 前処理と型合わせ
        hr = self._preprocess_horse_results(horse_results_df)

        # hr側の horse_id も保証
        if 'horse_id' not in hr.columns:
            if hr.index.name == 'horse_id' or 'horse_id' in hr.index.names:
                hr = hr.reset_index(level='horse_id')
            elif hr.index.name is None and not isinstance(hr.index, pd.MultiIndex):
                hr = hr.reset_index().rename(columns={'index': 'horse_id'})
        
        # IDを文字列に統一
        hr['horse_id'] = hr['horse_id'].astype(str)
        df['horse_id'] = df['horse_id'].astype(str)
        
        # 日付正規化
        hr['date'] = pd.to_datetime(hr['date'], errors='coerce').dt.normalize()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        
        # 数値化
        rank_col = '着順' if '着順' in hr.columns else '着 順'
        hr['rank_num'] = pd.to_numeric(hr[rank_col], errors='coerce')
        hr['is_win'] = (hr['rank_num'] == 1).astype(int)
        hr['is_place'] = (hr['rank_num'] <= 3).astype(int)
        hr['last_3f_num'] = pd.to_numeric(hr['上り'], errors='coerce') if '上り' in hr.columns else np.nan

        # 2. 特徴量計算 (Expanding Window + Shift(1))
        # 処理対象の馬のみに絞る
        target_horses = df['horse_id'].unique()
        hr_filtered = hr[hr['horse_id'].isin(target_horses)].copy()
        
        # df側に結果カラムをNaNで作成 (リーク防止)
        df_for_calc = df.copy()
        for col in ['rank_num', 'is_win', 'is_place', 'last_3f_num']:
            df_for_calc[col] = np.nan
        df_for_calc['_is_target'] = True
        hr_filtered['_is_target'] = False
        
        combined = pd.concat([hr_filtered, df_for_calc], sort=False).sort_values(['horse_id', 'date'])
        
        # 厳密に過去のみにする
        gb = combined.groupby('horse_id')
        combined['avg_rank'] = gb['rank_num'].transform(lambda x: x.expanding().mean().shift(1))
        combined['win_rate'] = gb['is_win'].transform(lambda x: x.expanding().mean().shift(1))
        combined['place_rate'] = gb['is_place'].transform(lambda x: x.expanding().mean().shift(1))
        combined['race_count'] = gb['rank_num'].transform(lambda x: x.expanding().count().shift(1))
        combined['prev_rank'] = gb['rank_num'].transform(lambda x: x.shift(1))
        combined['avg_last_3f'] = gb['last_3f_num'].transform(lambda x: x.expanding().mean().shift(1))
        combined['interval'] = gb['date'].transform(lambda x: x.diff().dt.days).fillna(20)

        # 3. マージ
        res_features = combined[combined['_is_target'] == True].copy()
        
        w_col = '枠番' if '枠番' in df.columns else '枠 番' if '枠 番' in df.columns else None
        u_col = '馬番' if '馬番' in df.columns else '馬 番' if '馬 番' in df.columns else None
        m_keys = ['horse_id', 'date']
        if w_col: m_keys.append(w_col)
        if u_col: m_keys.append(u_col)
        
        f_cols = m_keys + ['avg_rank', 'win_rate', 'place_rate', 'race_count', 'prev_rank', 'avg_last_3f', 'interval']
        res_features = res_features[f_cols].drop_duplicates(m_keys)
        
        df = df.merge(res_features, on=m_keys, how='left', suffixes=('', '_new'))
        
        for col in ['avg_rank', 'win_rate', 'place_rate', 'race_count', 'prev_rank', 'avg_last_3f', 'interval']:
            new_col = col + '_new'
            if new_col in df.columns:
                df[col] = df[new_col].fillna(default_values.get(col, 0))
                df.drop(columns=[new_col], inplace=True)
        
        return df
    
    def add_course_suitability_features(self, df: pd.DataFrame, horse_results: pd.DataFrame) -> pd.DataFrame:
        """
        コース適性（競馬場・距離・馬場）の特徴量を追加（超高速化版: 馬単位ベクトル演算 + 日付フィルタ）
        """
        if horse_results.empty:
            df['same_distance_win_rate'] = 0.0
            df['same_type_win_rate'] = 0.0
            return df
            
        # dateカラムの再保証 (もし消えていたら復元)
        if 'date' not in df.columns:
            try:
                rid_str = df.index.astype(str)
                years = pd.to_numeric(rid_str.str[:4], errors='coerce').fillna(2020).astype(int)
                base_dates = pd.to_datetime(years.astype(str) + '-01-01')
                # 簡易ランク
                df['temp_year'] = years
                df['temp_rid'] = rid_str
                df['date_rank'] = df.groupby('temp_year')['temp_rid'].rank(method='dense')
                df['date'] = base_dates + pd.to_timedelta(df['date_rank'], unit='D')
                df.drop(columns=['temp_year', 'temp_rid', 'date_rank'], inplace=True)
            except:
                pass
        
        # 1. horse_resultsの前処理（日付抽出など）
        hr = self._preprocess_horse_results(horse_results)
        
        # 2. 着順数値化
        if '着順' in hr.columns:
            hr['rank_num'] = pd.to_numeric(hr['着順'], errors='coerce')
        elif '着 順' in hr.columns:
            hr['rank_num'] = pd.to_numeric(hr['着 順'], errors='coerce')
        else:
            hr['rank_num'] = np.nan
        
        # 3. 距離抽出とタイプ判定
        def extract_dist(s):
            m = re.search(r'\d+', str(s))
            return int(m.group()) if m else 0
        
        if '距離' in hr.columns:
            hr['dist_num'] = hr['距離'].apply(extract_dist)
            hr['is_dirt'] = hr['距離'].astype(str).str.contains('ダ').astype(int)
        else:
            hr['dist_num'] = 0
            hr['is_dirt'] = 0
            
        # 4. horse_idの調整
        if 'horse_id' not in hr.columns:
            if hr.index.name == 'horse_id':
                hr = hr.reset_index()
            elif 'index' in hr.columns: # reset_index済みの場合
                hr = hr.rename(columns={'index': 'horse_id'})
            else:
                # インデックスをカラムにする
                hr = hr.reset_index()
                if 'index' in hr.columns:
                    hr = hr.rename(columns={'index': 'horse_id'})
        
        # 5. dfへの日付付与
        # マージキーの準備
        df_working = df.copy()
        
        # 以前はここで horse_results から日付をマージしていたが、
        # dfには既に date がある（冒頭で保証済み）ため、マージすると date_x, date_y になって date が消える。
        # したがってマージは不要。
        
        # ただし、以降のロジックで df_working['temp_iloc_idx'] などを使っているので、df_working は必要。
        
        # マージできなかった行（日付不明）は、現在時刻（未来）として扱うか、過去なしとして扱う
        # 安全側に倒して、日付不明なら「データなし」とするため、非常に古い日付を入れる手もあるが、
        # ここではfillnaせずにNaNのままにし、比較時にFalseになるようにする
        
        # 6. グループ化して計算
        hr_grouped = hr.groupby('horse_id')
        df_grouped = df_working.groupby('horse_id')
        
        # 結果格納用
        same_dist_rates = np.zeros(len(df))
        same_type_rates = np.zeros(len(df))
        
        # 元のインデックス復元用
        df_working['temp_iloc_idx'] = range(len(df_working))
        id_map = df_working[['horse_id', 'temp_iloc_idx']].set_index('horse_id')
        
        # 共通の馬IDのみ処理
        common_horses = set(df_grouped.groups.keys()) & set(hr_grouped.groups.keys())
        
        for hid in common_horses:
            # その馬の全レース履歴 (M行)
            hist = hr_grouped.get_group(hid)
            
            # その馬の今回のレース一覧 (K行)
            curr = df_grouped.get_group(hid)
            
            # ベクトル化のための準備
            # hist_dists: (M,) -> (1, M)
            hist_dists = hist['dist_num'].values.reshape(1, -1)
            hist_dirt = hist['is_dirt'].values.reshape(1, -1)
            hist_ranks = hist['rank_num'].values
            hist_date = hist['date'].values.reshape(1, -1)
            
            # curr_dists: (K,) -> (K, 1)
            curr_dists = curr['course_len'].fillna(2000).values.reshape(-1, 1)
            curr_types = curr['race_type'].fillna(0).values 
            curr_date = curr['date'].values.reshape(-1, 1)
            
            # 日付マスク: historyの日付 < currentの日付
            # curr_dateがNaTの場合、比較結果はFalse (安全)
            with np.errstate(invalid='ignore'): # NaT比較の警告抑制
                date_mask = (hist_date < curr_date)
            
            # 1. 距離判定 (Broadcasting)
            # abs(curr - hist) <= 100 AND date_mask
            dist_match = (np.abs(curr_dists - hist_dists) <= 100) & date_mask
            
            # 勝率計算
            # hist_ranks == 1 (wins) -> (M,) boolean
            wins_mask = (hist_ranks == 1)
            
            # Sum match counts along axis 1 (history axis) -> (K,)
            dist_counts = dist_match.sum(axis=1)
            dist_wins = (dist_match & wins_mask).sum(axis=1)
            
            # Zero division handling
            d_rates = np.divide(dist_wins, dist_counts, out=np.zeros_like(dist_wins, dtype=float), where=dist_counts!=0)
            
            # 2. タイプ判定
            # curr_is_dirt: (K,)
            curr_is_dirt = np.array([('ダ' in str(t) or t == 1) for t in curr_types]).reshape(-1, 1)
            
            # hist_dirt: (1, M)
            # match: (K, 1) == (1, M) -> (K, M)
            type_match = (curr_is_dirt == hist_dirt) & date_mask
            
            type_counts = type_match.sum(axis=1)
            type_wins = (type_match & wins_mask).sum(axis=1)
            
            t_rates = np.divide(type_wins, type_counts, out=np.zeros_like(type_wins, dtype=float), where=type_counts!=0)
            
            # 結果を元の位置に格納
            indices = curr['temp_iloc_idx'].values
            same_dist_rates[indices] = d_rates
            same_type_rates[indices] = t_rates
            
        # 列追加
        df['same_distance_win_rate'] = same_dist_rates
        df['same_type_win_rate'] = same_type_rates
        
        return df
    
    def add_jockey_features(self, df: pd.DataFrame, jockey_stats: pd.DataFrame = None) -> tuple:
        """
        騎手の特徴量を追加（リーク防止版: 時系列集計）
        Returns: (df, latest_jockey_stats)
        """
        df = df.copy()
        
        # 予測モード: 既存の統計データをマージ
        if jockey_stats is not None:
            # jockey_idが一致するものだけマージ
            if 'jockey_id' in df.columns:
                df = pd.merge(df, jockey_stats, on='jockey_id', how='left')
                
                # 欠損埋め（新規騎手など）
                defaults = {'jockey_avg_rank': 8.0, 'jockey_win_rate': 0.08, 'jockey_return_avg': 75.0}
                for col, val in defaults.items():
                    if col in df.columns:
                        df[col] = df[col].fillna(val)
            else:
                 df['jockey_avg_rank'] = 8.0
                 df['jockey_avg_rank'] = 8.0
                 df['jockey_win_rate'] = 0.08
                 df['jockey_return_avg'] = 75.0
            
            return df, jockey_stats

        # 学習モード: 時系列で特徴量生成
        if 'jockey_id' not in df.columns:
            return df, None
            
        # 必要なカラムの準備（Indexをリセットしてユニークにする）
        df_work = df.reset_index()
        
        # ソートキーの決定
        sort_cols = []
        if 'date' in df_work.columns:
            sort_cols = ['date', 'race_num']
        else:
            # dateがない場合はindex (race_id) を使用
            # race_idは時系列順になっていると仮定
            sort_cols = [df_work.index.name or 'index']
            if df_work.index.name is None:
                df_work.index.name = 'index'
        
        # ソート実行
        if 'date' in df_work.columns:
            df_work = df_work.sort_values(['date', 'race_num'], kind='mergesort')
        else:
            df_work = df_work.sort_index(kind='mergesort')
        
        # 着順数値化
        if '着順' in df_work.columns:
            df_work['rank_num'] = pd.to_numeric(df_work['着順'], errors='coerce')
        else:
            df_work['rank_num'] = np.nan
            
        df_work['is_win'] = (df_work['rank_num'] == 1).astype(int)
        
        # Groupby & Expanding
        # shift(1)することで、「前のレースまでの成績」にする（リーク回避）
        grouped = df_work.groupby('jockey_id')
        
        # 着順平均
        df_work['jockey_avg_rank'] = grouped['rank_num'].transform(lambda x: x.shift(1).expanding().mean())
        
        # 勝率
        # 勝率
        df_work['jockey_win_rate'] = grouped['is_win'].transform(lambda x: x.shift(1).expanding().mean())

        # 回収値 (平均払戻額)
        # 単勝オッズがある場合のみ
        if '単勝' in df_work.columns:
            odds = pd.to_numeric(df_work['単勝'], errors='coerce').fillna(0)
        elif '単 勝' in df_work.columns:
            odds = pd.to_numeric(df_work['単 勝'], errors='coerce').fillna(0)
        else:
            odds = 0
            
        # 払い戻し額 (100円賭けた場合)
        # is_win(0/1) * odds * 100
        df_work['return'] = df_work['is_win'] * odds * 100
        
        # 平均回収値 (expanding mean)
        df_work['jockey_return_avg'] = grouped['return'].transform(lambda x: x.shift(1).expanding().mean())
        df_work['jockey_return_avg'] = df_work['jockey_return_avg'].fillna(75.0) # 平均的な控除率後の値
        
        # 欠損値（初騎乗時など）を埋める
        df_work['jockey_avg_rank'] = df_work['jockey_avg_rank'].fillna(8.0)
        df_work['jockey_win_rate'] = df_work['jockey_win_rate'].fillna(0.08)
        
        # 元の順序に戻すためにindexを使用 (reset_indexしたのでsort_indexで戻る)
        df_work = df_work.sort_index()
        
        # dfに反映 (dfはまだ元のIndexのままなので、df_workの値を代入するには注意が必要)
        # ここでは df_work をそのまま返してしまうのが安全 (reset_indexされた状態)
        # ただし元のIndex (race_id) が消えると困る場合があるので set_index する
        
        # 元のIndex名を取得
        original_index_name = df.index.name
        
        # マージや代入ではなく、df_work自体を採用する
        # ただし、呼び出し元でIndexを期待しているかもしれない
        
        if original_index_name:
            df_work = df_work.set_index(original_index_name)
        else:
            # 元のIndexが 'index' という名前でカラムに残っているはず
            if 'index' in df_work.columns:
                 df_work = df_work.set_index('index')
                 df_work.index.name = None # 元が名無しならNoneに
            
        df = df_work
        
        # 最新の成績（全期間平均）を作成して返す（予測用）
        # 各騎手の全期間の平均を計算
        final_rank = df_work.groupby('jockey_id')['rank_num'].mean()
        final_win = df_work.groupby('jockey_id')['is_win'].mean()
        
        latest_stats = pd.DataFrame({
            'jockey_id': final_rank.index,
            'jockey_avg_rank': final_rank.values,
            'jockey_win_rate': final_win.values,
            'jockey_return_avg': df_work.groupby('jockey_id')['return'].mean().values
        })
        
        return df, latest_stats
    
    def add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        オッズと人気を特徴量として追加
        """
        df = df.copy()
        
        # 単勝オッズ
        if '単勝' in df.columns:
            df['odds'] = pd.to_numeric(df['単勝'], errors='coerce')
        elif '単 勝' in df.columns:
            df['odds'] = pd.to_numeric(df['単 勝'], errors='coerce')
        else:
            df['odds'] = np.nan
            
        # 人気
        if '人気' in df.columns:
            df['popularity'] = pd.to_numeric(df['人気'], errors='coerce')
        elif '人 気' in df.columns:
            # データによってスペースが入っている場合
            df['popularity'] = pd.to_numeric(df['人 気'], errors='coerce')
        else:
            df['popularity'] = np.nan
            
        # 欠損補完 (学習時は平均、予測時は直前オッズがない場合に備える)
        # ここではとりあえず中央値や平均で埋めるが、予測精度に影響する
        # 学習データには基本的にあるはず。
        df['odds'] = df['odds'].fillna(100.0) # 欠損は超大穴扱い
        df['popularity'] = df['popularity'].fillna(10) # 欠損は下位人気
        
        return df
    
    def add_pedigree_features(self, df: pd.DataFrame, peds_df: pd.DataFrame) -> pd.DataFrame:
        """
        血統の特徴量を追加（高速化版）
        """
        df = df.copy()
        
        # 血統スコア定義 (簡易版)
        ped_scores = {
            'speed': ['ディープインパクト', 'ロードカナロア', 'サクラバクシンオー', 'ダイワメジャー', 'キングカメハメハ'],
            'stamina': ['ハーツクライ', 'オルフェーヴル', 'ゴールドシップ', 'ステイゴールド', 'エピファネイア'],
            'dirt': ['ヘニーヒューズ', 'シニスターミニスター', 'ゴールドアリュール', 'パイロ', 'クロフネ']
        }
        
        if peds_df.empty:
            for cat in ped_scores.keys():
                df[f'peds_score_{cat}'] = 0
            return df

        # peds_dfに既にスコアカラムがある場合は計算をスキップ (高速化)
        precomputed = True
        for cat in ped_scores.keys():
            if f'peds_score_{cat}' not in peds_df.columns:
                precomputed = False
                break
        
        if precomputed:
             scores_df = peds_df
        else:
            # peds_df側で一括計算
            # 文字列結合 (高速化のため values を使用)
            try:
                peds_str_series = peds_df.fillna('').astype(str).agg(' '.join, axis=1)
            except Exception:
                 # aggが失敗する場合はapplyで
                 peds_str_series = peds_df.fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
            
            scores_dict = {}
            for cat, sires in ped_scores.items():
                # count occurrences
                def count_s(text):
                    c = 0
                    for s in sires:
                        if s in text: c += 1
                    return c
                scores_dict[f'peds_score_{cat}'] = peds_str_series.apply(lambda x: count_s(x))
                
            scores_df = pd.DataFrame(scores_dict, index=peds_df.index)

        # mapでdfに結合
        for cat in ped_scores.keys():
            col = f'peds_score_{cat}'
            # マッピング、該当なしは0
            df[col] = df['horse_id'].map(scores_df[col]).fillna(0).astype(int)
                    
        # 既存のIDマップ
        if 0 in peds_df.columns:
            df['sire'] = df['horse_id'].map(peds_df[0])
        if 1 in peds_df.columns:
            df['dam'] = df['horse_id'].map(peds_df[1])
        
        return df
    
    def create_target(self, df: pd.DataFrame, target_type: str = 'rank') -> pd.DataFrame:
        """
        目的変数を作成
        
        Args:
            df: DataFrame
            target_type: 目的変数のタイプ ('rank', 'win', 'place')
            
        Returns:
            目的変数追加済みのDataFrame
        """
        df = df.copy()
        
        if target_type == 'rank':
            # 着順をそのまま使用（小さいほど良い）
            df['target'] = df['着順']
        elif target_type == 'win':
            # 勝利フラグ（1着かどうか）
            df['target'] = (df['着順'] == 1).astype(int)
        elif target_type == 'place':
            # 複勝フラグ（3着以内かどうか）
            df['target'] = (df['着順'] <= 3).astype(int)
        
        return df

    def create_bias_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        トラックバイアス（枠順別成績）マップを作成
        競馬場 x 芝ダ x 距離区分 x 枠番 ごとの成績を集計
        """
        df = df.copy()
        
        # 距離区分 (SMILE区分: S<1300, M<1900, I<2100, L<2700, E>=2700)
        df['dist_bin'] = pd.cut(df['course_len'], 
                                bins=[0, 1300, 1899, 2100, 2700, 9999],
                                labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 3着以内率（複勝率）を集計
        if '着順' in df.columns:
            df['is_place'] = (pd.to_numeric(df['着順'], errors='coerce') <= 3).astype(int)
        else:
            return pd.DataFrame()
            
        # 集計キー
        keys = ['venue_id', 'race_type', 'dist_bin', '枠番']
        
        # グループ化して平均算出
        bias_map = df.groupby(keys)['is_place'].mean().reset_index()
        bias_map.rename(columns={'is_place': 'waku_bias_rate'}, inplace=True)
        
        return bias_map
    
    def add_bias_features(self, df: pd.DataFrame, bias_map: pd.DataFrame) -> pd.DataFrame:
        """
        トラックバイアス特徴量を追加
        """
        if bias_map is None or bias_map.empty:
            df['waku_bias_rate'] = 0.3 # default
            return df
            
        df = df.copy()
        
        # 距離区分作成 (SMILE区分)
        df['dist_bin'] = pd.cut(df['course_len'], 
                                bins=[0, 1300, 1899, 2100, 2700, 9999],
                                labels=[0, 1, 2, 3, 4]).astype(int)
        
        # マージ
        keys = ['venue_id', 'race_type', 'dist_bin', '枠番']
        df = pd.merge(df, bias_map, on=keys, how='left')
        
        # 欠損埋め（平均値または0.3付近）
        df['waku_bias_rate'] = df['waku_bias_rate'].fillna(df['waku_bias_rate'].mean()).fillna(0.3)
        
        # 一時カラム削除
        if 'dist_bin' in df.columns:
            df.drop(columns=['dist_bin'], inplace=True)
            
        return df


def prepare_training_data(results_df: pd.DataFrame, 
                          horse_results_df: pd.DataFrame = None,
                          peds_df: pd.DataFrame = None,
                          scale: bool = False) -> tuple:
    """
    学習用データを準備
    
    Args:
        results_df: レース結果DataFrame
        horse_results_df: 馬の過去成績DataFrame
        peds_df: 血統DataFrame
        
    Returns:
        (特徴量DataFrame, 目的変数Series, DataProcessor, FeatureEngineer)
    """
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 基本的な前処理
    df = processor.process_results(results_df)
    
    # dateカラムの保証 (ない場合はrace_idから擬似生成)
    if 'date' not in df.columns:
        print("Warning: 'date' column missing. Generating pseudo-date from race_id.")
        try:
            # race_id: YYYY TT KK DD RR (12桁)
            # YYYY:0-4, TT:4-6, KK:6-8, DD:8-10, RR:10-12
            rid_str = df.index.astype(str)
            
            # 年
            years = pd.to_numeric(rid_str.str[:4], errors='coerce').fillna(2020).astype(int)
            # 開催回・変数 (時系列順序用) - 簡易的に数値化して日数として足す
            # 1回=8日程度、場所コードなどもあるが、単純に辞書順が時系列と仮定
            # 下8桁を数値化して、それを「秒」や「分」として加算してもいいが、daysとして加算するには大きすぎる
            
            # 擬似日付: Year-01-01 + (Indexの順序)
            # Indexがソートされていればこれで正しい順序になる
            # race_idは time series ordered であると仮定
            
            # 擬似日付: Year-01-01 + (Indexの刻み)
            # 日数として足すと年を越えてしまうため、分単位で足して同一年に留める
            base_dates = pd.to_datetime(years.astype(str) + '-01-01')
            
            # 年ごとにグループ化してランク付け（同一レース内の馬は同じランクになるよう rank_ids を使う）
            df['temp_year'] = years
            df['temp_rid'] = rid_str
            df['date_rank'] = df.groupby('temp_year')['temp_rid'].rank(method='dense')
            
            # date = Year-01-01 + date_rank (minutes)
            # これにより、一日のうちに全レースが収まり、年は race_id と一致する
            df['date'] = base_dates + pd.to_timedelta(df['date_rank'], unit='min')
            
            # 後始末
            df.drop(columns=['temp_year', 'temp_rid', 'date_rank'], inplace=True)
            
        except Exception as e:
            print(f"Error generating pseudo-date: {e}")
            # 最悪の場合、全て特定の日付にする（機能は落ちるがエラーは回避）
            df['date'] = pd.to_datetime('2020-01-01')
    
    # 馬の過去成績特徴量
    if horse_results_df is not None and not horse_results_df.empty:
        # カラム名の正規化と型変換
        horse_results_df = horse_results_df.copy()
        horse_results_df.columns = horse_results_df.columns.str.replace(' ', '')
        if '着順' in horse_results_df.columns:
            horse_results_df['着順'] = pd.to_numeric(horse_results_df['着順'], errors='coerce')
            
        df = engineer.add_horse_history_features(df, horse_results_df)
        df = engineer.add_course_suitability_features(df, horse_results_df)
    
    # 騎手特徴量
    df, jockey_stats = engineer.add_jockey_features(df)
    
    # 血統特徴量
    if peds_df is not None and not peds_df.empty:
        df = engineer.add_pedigree_features(df, peds_df)
    
    # 目的変数作成
    df = engineer.create_target(df, target_type='place')

    # トラックバイアス特徴量 (枠順成績) の追加
    # 学習データ全体を使って Bias Map を作成
    bias_map = engineer.create_bias_map(df)
    df = engineer.add_bias_features(df, bias_map)
    
    # オッズ・人気特徴量 (NEW)
    df = engineer.add_odds_features(df)
    
    # カテゴリ変数のエンコード
    categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    df = processor.encode_categorical(df, categorical_cols)
    
    # 特徴量カラムを選択
    feature_cols = [
        '枠番', '馬番', '斤量', '年齢',
        '体重', '体重変化', 'course_len',
        'avg_rank', 'win_rate', 'place_rate', 'race_count',
        'jockey_avg_rank', 'jockey_win_rate', 'jockey_return_avg', # Added
        'venue_id', 'kai', 'day', 'race_num',
        'avg_last_3f', 'avg_running_style',
        'interval', 'prev_rank',
        'same_distance_win_rate', 'same_type_win_rate', # Restored
        'peds_score_speed', 'peds_score_stamina', 'peds_score_dirt',
        'waku_bias_rate',
        'odds', 'popularity', # Added
        'original_race_id' # Added for ID restoration
    ]
    feature_cols.extend(categorical_cols)
    
    # 存在するカラムのみ使用
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # 欠損値を埋める (数値カラムのみ)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    if scale:
        processor.fit_scale(X)
        X = processor.transform_scale(X)
    
    return X, y, processor, engineer, bias_map, jockey_stats, df
