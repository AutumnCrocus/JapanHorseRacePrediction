"""
複合馬券（馬連、馬単、ワイド、三連複、三連単）の戦略モジュール
"""
import pandas as pd
import numpy as np
import itertools

class CompositeBettingStrategy:
    """複合馬券戦略クラス"""
    
    @staticmethod
    def generate_box_bets(df_pred, n_horses=5, bet_types=['馬連']):
        """
        AIスコア上位馬のボックス買い目を生成する
        
        Args:
            df_pred: 予測結果DataFrame (score, horse_number必須)
            n_horses: ボックスに含める頭数 (上位N頭)
            bet_types: 生成する券種リスト ('馬連', '馬単', 'ワイド', '3連複', '3連単')
            
        Returns:
            list of dict: 買い目リスト
        """
        # Debug
        if 'horse_number' not in df_pred.columns:
            print(f"DEBUG: Missing horse_number! Cols: {df_pred.columns}, Head: {df_pred.head(1)}")
            
        # Scoreでソートして上位N頭を抽出
        top_n = df_pred.sort_values('score', ascending=False).head(n_horses)
        
        if len(top_n) < 2:
            return []
            
        horse_nums = top_n['horse_number'].tolist()
        horse_nums.sort() # 組み合わせのためにソート（単勝以外）
        
        bets = []
        
        # 1. 馬連 (Uma-ren): 2頭選ぶ、順序なし
        if '馬連' in bet_types:
            combos = list(itertools.combinations(horse_nums, 2))
            for c in combos:
                bets.append({
                    'type': '馬連',
                    'combo': f"{c[0]}-{c[1]}", # netkeiba形式: 小さい順
                    'amount': 100
                })
                
        # 2. ワイド (Wide): 2頭選ぶ、順序なし
        if 'ワイド' in bet_types:
            combos = list(itertools.combinations(horse_nums, 2))
            for c in combos:
                bets.append({
                    'type': 'ワイド',
                    'combo': f"{c[0]}-{c[1]}",
                    'amount': 100
                })

        # 3. 馬単 (Uma-tan): 2頭選ぶ、順序あり
        if '馬単' in bet_types:
            # permutations (順序あり)
            perms = list(itertools.permutations(horse_nums, 2))
            for p in perms:
                bets.append({
                    'type': '馬単',
                    'combo': f"{p[0]}→{p[1]}", # 1着→2着 (netkeiba format)
                    'amount': 100
                })
                
        # 4. 3連複 (Sanren-puku): 3頭選ぶ、順序なし
        if '3連複' in bet_types and len(horse_nums) >= 3:
            combos = list(itertools.combinations(horse_nums, 3))
            for c in combos:
                # ソート済みなのでそのまま結合
                bets.append({
                    'type': '3連複',
                    'combo': f"{c[0]}-{c[1]}-{c[2]}",
                    'amount': 100
                })

        # 5. 3連単 (Sanren-tan): 3頭選ぶ、順序あり
        if '3連単' in bet_types and len(horse_nums) >= 3:
            perms = list(itertools.permutations(horse_nums, 3))
            for p in perms:
                bets.append({
                    'type': '3連単',
                    'combo': f"{p[0]}→{p[1]}→{p[2]}", # Arrow format
                    'amount': 100
                })
                
        return bets

    @staticmethod
    def calculate_return(bets, return_df):
        """
        買い目と払い戻しデータを照合して回収額を計算
        
        Args:
            bets: 買い目リスト
            return_df: そのレースの払い戻しデータ (df[0]==券種, df[1]==組み合わせ, df[2]==金額)
            
        Returns:
            int: 払い戻し総額
        """
        total_return = 0
        
        # 払い戻しデータを辞書化して高速検索
        # key: (type, combo), value: list of return amounts (ワイドなどは複数あるため)
        payout_map = {}
        
        for _, row in return_df.iterrows():
            b_type = row[0]
            combo = str(row[1])
            try:
                # "1,200円" -> 1200
                amt = int(str(row[2]).replace(',', '').replace('円', ''))
            except:
                amt = 0
                
            key = (b_type, combo)
            if key not in payout_map:
                payout_map[key] = []
            payout_map[key].append(amt)
            
        # 照合
        for bet in bets:
            b_type = bet['type']
            combo = bet['combo']
            
            # ワイドの normalize (netkeiba形式: 小さい順) is handled in generation?
            # 組み合わせ馬券(馬連, 3連複, ワイド)は番号順にソートされている必要がある
            # 馬単, 3連単は順序通り
            
            key = (b_type, combo)
            if key in payout_map:
                # 的中
                # 払い戻し額を加算 (賭け金に応じて比例配分)
                # bet['amount'] が 500円なら、配当(100円あたり)の5倍
                amount = bet.get('amount', 100)
                factor = amount / 100
                
                # 万が一、同着などで複数の払い戻しがある場合もリストから合算
                base_payout = sum(payout_map[key])
                total_return += base_payout * factor
                
        return total_return
