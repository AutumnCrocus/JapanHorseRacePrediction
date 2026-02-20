
import pandas as pd
import numpy as np

class ExperimentalStrategies:
    """
    実験的な馬券戦略を定義するクラス
    """

    @staticmethod
    def dynamic_box(df_preds: pd.DataFrame, budget: int, odds_data: dict = None) -> list:
        """
        Dynamic Box Strategy (Refined):
        累積確率(Cumulative Probability)に基づいてBOXの頭数を動的に決定する。
        - 確率を正規化(Softmax/Normalize)してから累積和を計算。
        - 閾値(0.85)を超えるまで採用。
        """
        recommendations = []
        if df_preds.empty:
            return recommendations

        # 設定 (Relaxed)
        CUMULATIVE_THRESHOLD = 0.85
        MAX_HORSES = 6 # 3連複BOX(20点=2000円)まで
        MIN_HORSES = 4 # 最低4頭は拾う (4点=400円)

        df_sorted = df_preds.sort_values('probability', ascending=False).copy()
        
        # 確率の正規化 (単純な割合に変換)
        probs = df_sorted['probability'].values
        if probs.sum() > 0:
            df_sorted['norm_prob'] = probs / probs.sum()
        else:
            df_sorted['norm_prob'] = 1.0 / len(probs)
            
        df_sorted['cum_prob'] = df_sorted['norm_prob'].cumsum()
        
        # 閾値を超えるまでの馬を選択
        cutoff_row = 0
        df_reset = df_sorted.reset_index(drop=True)
        # 閾値を超えた最初のインデックスを取得
        hits = df_reset[df_reset['cum_prob'] >= CUMULATIVE_THRESHOLD]
        if not hits.empty:
            cutoff_row = hits.index[0]
            count = cutoff_row + 1
        else:
            count = len(df_sorted) # 全頭でも届かない場合(ありえないが)
            
        # 制約適用 (Min保証 -> Maxキャップ)
        count = max(MIN_HORSES, count)
        count = min(count, MAX_HORSES)
        
        # 有効な馬番号のみ抽出
        selected_horses = df_sorted.iloc[:count]['horse_number'].dropna().astype(int).tolist()
        
        # 券種選択 (予算内で買える最も配当性の高い券種を選ぶロジック)
        # 3連複 > 馬連 > ワイド
        
        # 予算に収まるように頭数を調整 (Adaptive Reduction)
        n = len(selected_horses)
        best_bet = None
        
        # Try finding a valid bet by reducing n if necessary
        curr_n = n
        while curr_n >= MIN_HORSES:
            # Check 3-renpuku
            if curr_n >= 3:
                cost_sanren = (curr_n * (curr_n-1) * (curr_n-2) // 6) * 100
                if cost_sanren <= budget:
                    best_bet = {
                        'type': '3連複', 
                        'n': curr_n, 
                        'cost': cost_sanren, 
                        'points': curr_n * (curr_n-1) * (curr_n-2) // 6
                    }
                    break
            
            # Check Umaren (If 3-renpuku not possible or budget tight?)
            # Usually we prefer 3-renpuku for coverage, but if n=valid for Umaren and not 3-renpuku?
            # Actually, cost of nC3 is always < nC2? No. 
            # 4C3=4, 4C2=6. 5C3=10, 5C2=10. 6C3=20, 6C2=15.
            # At n=6, 3ren costs 2000, Umaren costs 1500.
            # So if budget=1500, we can buy Umaren 6Box but not 3Ren 6Box.
            
            if curr_n >= 2:
                cost_umaren = (curr_n * (curr_n-1) // 2) * 100
                if cost_umaren <= budget:
                    # Preference check: Is Umaren(n) better than 3Ren(n-1)?
                    # For now, prioritize larger n (Coverage)
                    if best_bet is None: # Found first valid
                        best_bet = {
                            'type': '馬連', 
                            'n': curr_n, 
                            'cost': cost_umaren, 
                            'points': curr_n * (curr_n-1) // 2
                        }
                        # Don't break immediately, check if 3ren with n-1 is possible? 
                        # No, simpler logic: Prefer 3ren if possible at current n. If not, try Umaren. 
                        # If neither, reduce n.
                        break
            
            curr_n -= 1
            
        if best_bet:
            # Re-select horses based on final n
            final_horses = selected_horses[:best_bet['n']]
            rec = {
                'bet_type': best_bet['type'],
                'method': 'BOX',
                'horse_numbers': final_horses,
                'formation': [final_horses],
                'combination': f"{','.join(map(str, final_horses))} BOX",
                'unit_amount': 100,
                'total_amount': best_bet['cost'],
                'points': best_bet['points'],
                'desc': f'DynamicBox(n={best_bet["n"]}, {best_bet["type"]})',
                'reason': f'累積確率カバー(BudgetAdjusted)'
            }
            recommendations.append(rec)
            
        return recommendations

    @staticmethod
    def value_hunter(df_preds: pd.DataFrame, budget: int, odds_data: dict = None) -> list:
        """
        Value Hunter Strategy:
        AIの評価が高い(Top 5以内かつProb > 10%)が、オッズが美味しい(単勝10倍以上)馬を狙い撃つ。
        - 該当馬がいる場合: 単勝 + 複勝 + ワイド流し(軸:穴馬, 相手:上位人気)
        """
        recommendations = []
        if df_preds.empty:
            return recommendations
            
        if not odds_data or 'tan' not in odds_data:
            return recommendations
            
        tan_odds = odds_data['tan']
        
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top_horses = df_sorted.head(5) # 上位5頭をチェック
        
        target_holes = []
        popular_horses = [] # 相手用の人気馬
        
        for _, row in df_sorted.iterrows():
            h_num = int(row['horse_number'])
            prob = row['probability']
            odds = row.get('odds', 0)
            
            # オッズデータ補完
            if odds == 0 and h_num in tan_odds:
                odds = tan_odds[h_num]
                
            # 穴馬条件: 上位評価 & オッズ10倍以上 & 確率10%以上
            if odds >= 10.0 and prob >= 0.10:
                target_holes.append(h_num)
            
            # 人気馬（相手）条件: オッズ5倍未満 または 確率20%以上
            if (odds < 5.0 or prob >= 0.20) and h_num not in target_holes:
                popular_horses.append(h_num)
                
        # 予算配分
        remaining = budget
        
        # 穴馬が見つかった場合のみ実行
        for hole in target_holes:
            if remaining < 100: break
            
            # 1. 単勝 (予算の20%)
            win_amount = (int(budget * 0.2) // 100) * 100
            if win_amount > 0 and remaining >= win_amount:
                recommendations.append({
                    'bet_type': '単勝', 'method': 'SINGLE', 'horse_numbers': [hole],
                    'unit_amount': win_amount, 'total_amount': win_amount, 'points': 1,
                    'desc': 'ValueHunter(Win)', 'combination': str(hole),
                    'reason': f'穴馬狙い(オッズ{tan_odds.get(hole)}倍)'
                })
                remaining -= win_amount
                
            # 2. 複勝 (予算の30%)
            place_amount = (int(budget * 0.3) // 100) * 100
            if place_amount > 0 and remaining >= place_amount:
                recommendations.append({
                    'bet_type': '複勝', 'method': 'SINGLE', 'horse_numbers': [hole],
                    'unit_amount': place_amount, 'total_amount': place_amount, 'points': 1,
                    'desc': 'ValueHunter(Place)', 'combination': str(hole),
                    'reason': '保険'
                })
                remaining -= place_amount
                
            # 3. ワイド流し (残り予算で、相手は人気馬へ)
            if remaining >= 100 and popular_horses:
                # 相手は最大3頭まで
                opponents = popular_horses[:3]
                points = len(opponents)
                unit_wide = (remaining // points // 100) * 100
                if unit_wide >= 100:
                    cost = unit_wide * points
                    recommendations.append({
                        'bet_type': 'ワイド', 'method': '流し', 
                        'horse_numbers': [hole] + opponents,
                        'formation': [[hole], opponents],
                        'unit_amount': unit_wide, 'total_amount': cost, 'points': points,
                        'desc': 'ValueHunter(Wide)', 
                        'combination': f"軸:{hole}-相手:{','.join(map(str, opponents))}",
                        'reason': '穴軸流し'
                    })
                    remaining -= cost
        
        return recommendations

    @staticmethod
    def confidence_scaler(df_preds: pd.DataFrame, budget: int, odds_data: dict = None) -> list:
        """
        Confidence Scaler Strategy:
        確信度(Confidence)が高い場合にのみ大きく張り、低い場合は見送るメリハリ型。
        - 1位の確率が50%超: 単勝に全ツッパ(近い形)
        - 1位の確率が30%超: 馬連/馬単ながし
        - それ以外: 見送り (return empty) -> 結果としてRecovery Rateが高くなることを狙う
        """
        recommendations = []
        if df_preds.empty:
            return recommendations
            
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top_horse = df_sorted.iloc[0]
        prob_1st = top_horse['probability']
        h_1st = int(top_horse['horse_number'])
        
        # Case A: 圧倒的本命 (Prob >= 50%)
        if prob_1st >= 0.50:
            # 予算の80%を単勝、20%を馬単流し
            amt_win = (int(budget * 0.8) // 100) * 100
            if amt_win > 0:
                recommendations.append({
                    'bet_type': '単勝', 'method': 'SINGLE', 'horse_numbers': [h_1st],
                    'unit_amount': amt_win, 'total_amount': amt_win, 'points': 1,
                    'desc': 'ConfScaler(Winner)', 'combination': str(h_1st),
                    'reason': f'確信度S({prob_1st:.1%})'
                })
                
            amt_rem = budget - amt_win
            opponents = df_sorted.iloc[1:4]['horse_number'].tolist() # 2-4位へ流す
            if amt_rem >= len(opponents) * 100:
                unit = (amt_rem // len(opponents) // 100) * 100
                if unit > 0:
                    recommendations.append({
                        'bet_type': '馬単', 'method': '流し', 
                        'horse_numbers': [h_1st] + opponents,
                        'formation': [[h_1st], opponents],
                        'unit_amount': unit, 'total_amount': unit * len(opponents), 'points': len(opponents),
                        'desc': 'ConfScaler(Umatan)', 
                        'combination': f"軸:{h_1st}-相手:{','.join(map(str, opponents))}",
                        'reason': 'ボーナス狙い'
                    })
                    
        # Case B: 有力本命 (Prob >= 30%)
        elif prob_1st >= 0.30:
            # 馬連流し中心
            opponents = df_sorted.iloc[1:6]['horse_number'].tolist() # 2-6位へ流す(5点)
            cost = 500 # 最低500円
            
            if budget >= cost:
                unit = (budget // 5 // 100) * 100
                total = unit * 5
                recommendations.append({
                    'bet_type': '馬連', 'method': '流し', 
                    'horse_numbers': [h_1st] + opponents,
                    'formation': [[h_1st], opponents],
                    'unit_amount': unit, 'total_amount': total, 'points': 5,
                    'desc': 'ConfScaler(Umaren)', 
                    'combination': f"軸:{h_1st}-相手:{','.join(map(str, opponents))}",
                    'reason': f'確信度A({prob_1st:.1%})'
                })
        
        # Case C: 混戦 -> 見送り (return [])
        
        return recommendations
