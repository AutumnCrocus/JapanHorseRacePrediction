import pandas as pd
import numpy as np

class TrackBiasAnalyzer:
    """
    当日のトラックバイアス（枠順・脚質）を分析するクラス
    """
    
    def __init__(self):
        pass

    def analyze_bias(self, race_date: pd.Timestamp, venue_id: str, current_race_no: int, df_today_results: pd.DataFrame) -> dict:
        """
        指定されたレース以前の当日結果からバイアスを抽出する
        
        Args:
            race_date: レース日付
            venue_id: 開催場所ID (例: '05' 東京)
            current_race_no: 現在のレース番号 (これより前のレースを集計対象とする)
            df_today_results: 当日の全レース結果DataFrame (着順, 枠番, 通過順位などが必要)
            
        Returns:
            dict: {
                'frame_bias': 'inner' | 'outer' | 'flat',
                'position_bias': 'front' | 'back' | 'flat',
                'frame_score': float,  # 内枠有利度 (正:内, 負:外)
                'position_score': float # 先行有利度 (正:前, 負:後)
            }
        """
        # 当日の同会場・現在レース以前のデータをフィルタリング
        # df_today_results は既に日付と会場で絞り込まれている前提だが、念のため確認
        target_races = df_today_results[
            (df_today_results['race_no'] < current_race_no) &
            (df_today_results['着順'].isin([1, 2, 3])) # 3着以内を対象
        ].copy()
        
        if len(target_races) == 0:
            return {
                'frame_bias': 'flat',
                'position_bias': 'flat',
                'frame_score': 0.0,
                'position_score': 0.0
            }

        # === 枠順バイアス (Frame Bias) ===
        # 1-4枠を「内」、5-8枠を「外」として、3着内入線率を比較... ではなく
        # シンプルに3着内馬の平均枠番を見る
        # 平均が4.5より小さければ内有利、大きければ外有利
        avg_frame = target_races['枠番'].mean()
        # 1~8の中央は4.5
        # 3.5以下なら明確に内、5.5以上なら明確に外
        frame_score = 4.5 - avg_frame # 正なら内有利
        
        if frame_score > 0.5:
            frame_bias = 'inner'
        elif frame_score < -0.5:
            frame_bias = 'outer'
        else:
            frame_bias = 'flat'

        # === 脚質バイアス (Position Bias) ===
        # 4コーナー通過順位を使用
        # 通過順位を抽出 (例: "2-2-3" -> 3, "10-11" -> 11)
        # 最後の通過順位（4角）を取得
        def get_last_passing_rank(s):
            if not isinstance(s, str): return np.nan
            try:
                parts = s.split('-')
                return int(parts[-1])
            except:
                return np.nan

        target_races['last_passing'] = target_races['通過'].apply(get_last_passing_rank)
        
        # 3着内馬の平均4角通過順位
        # ただし頭数によって意味が変わるため、相対位置（順位/頭数）が良いが、
        # 簡易的に「5番手以内」率を見る
        front_runners = target_races[target_races['last_passing'] <= 5]
        front_ratio = len(front_runners) / len(target_races)
        
        # 基準: 3着内馬の50%以上が4角5番手以内なら先行有利
        # 70%以上なら圧倒的前有利
        position_score = front_ratio - 0.5 # 正なら前有利
        
        if front_ratio >= 0.6:
            position_bias = 'front'
        elif front_ratio <= 0.3: # 差しが決まっている
            position_bias = 'back'
        else:
            position_bias = 'flat'
            
        return {
            'frame_bias': frame_bias,
            'position_bias': position_bias,
            'frame_score': frame_score,
            'position_score': position_score,
            'sample_count': len(target_races) // 3 # レース数
        }
