"""
予算配分・買い目生成モジュール
予算に応じて最適な券種ポートフォリオ（3連単BOX、3連複BOX、単勝など）を構築する。
"""

import pandas as pd
import numpy as np
from .strategy_composite import CompositeBettingStrategy

class BettingAllocator:
    """
    予算配分を行うクラス
    """
    
    @staticmethod
    def allocate_budget(df_preds, budget: int, odds_data: dict = None, allowed_types: list = None) -> list:
        """
        予算に応じて推奨買い目を生成する
        
        Args:
            df_preds: 予測結果DataFrame (probability, horse_number, etc.)
            budget: 予算総額 (円)
            odds_data: オッズデータ (Optional)
            allowed_types: 許可する券種リスト (例: ['単勝'], ['3連単'])。Noneの場合は予算に応じた最適ミックス。
            
        Returns:
            list: 推奨買い目のリスト
        """
        recommendations = []
        
        # 予測確率順にソート（この順序でBOX候補を選ぶ）
        df_sorted = df_preds.sort_values('probability', ascending=False).copy()
        top_horses = df_sorted['horse_number'].tolist()
        
        if not top_horses:
            return []

        # --- Single Type Constraint Logic ---
        # ユーザー指定の券種のみで構成する場合
        if allowed_types and len(allowed_types) == 1:
            bet_type = allowed_types[0]
            
            rec = None
            
            if bet_type in ['単勝', '複勝']:
                # 一点買い (予算全額)
                # 最も自信のある馬に全額
                target_horse = top_horses[0]
                amount = (budget // 100) * 100 # 100円単位
                
                if amount > 0:
                    rec = {
                        'type': bet_type,
                        'method': 'SINGLE',
                        'formation': [[target_horse]],
                        'amount': amount,
                        'count': 1,
                        'desc': '一点買い',
                        'horses': [target_horse]
                    }
                    
            elif bet_type in ['馬連', 'ワイド', '3連複', '3連単', '馬単', '枠連']:
                # BOX買い
                # 予算内で買える最大頭数を計算
                # 馬連/ワイド/枠連/馬単(Boxは同じ点数ではないが): nC2 or nP2
                # 3連複: nC3
                # 3連単: nP3
                
                max_horses = 0
                points = 0
                
                # 計算ロジック
                # n頭BOXの点数を計算し、budget内か判定
                import math
                
                n_limit = min(len(top_horses), 18) # 最大18頭
                
                best_n = 0
                best_cost = 0
                best_points = 0
                
                for n in range(1, n_limit + 1):
                    p = 0
                    if bet_type in ['馬連', 'ワイド', '枠連']:
                        if n < 2: continue
                        p = n * (n - 1) // 2
                    elif bet_type == '馬単':
                        if n < 2: continue
                        p = n * (n - 1)
                    elif bet_type == '3連複':
                        if n < 3: continue
                        p = n * (n - 1) * (n - 2) // 6
                    elif bet_type == '3連単':
                        if n < 3: continue
                        p = n * (n - 1) * (n - 2)
                        
                    cost = p * 100
                    if cost <= budget:
                        best_n = n
                        best_cost = cost
                        best_points = p
                    else:
                        break
                
                if best_n > 0:
                    # BOX作成
                    # 特定の予算(500円など)で、Boxが組めない場合(3連単Box最小600円)はどうするか？
                    # -> 買えない (best_n=0のまま) -> recommendationsなし
                    
                    cand = top_horses[:best_n]
                    rec = {
                        'type': bet_type,
                        'method': 'BOX',
                        'formation': [cand],
                        'amount': best_cost,
                        'count': best_points,
                        'desc': f'{best_n}頭BOX',
                        'horses': cand
                    }
                    
            if rec:
                recommendations.append(rec)
                
            # フォーマット変換して返却
            return BettingAllocator._format_recommendations(recommendations)


        # --- Default Portfolio Logic (Mixed) ---
        remaining_budget = budget
        
        # ... (Existing logic for mixed portfolio) ...
        # (Since we are replacing the method, we must copy the existing logic or reference it)
        # For brevity in this tool call, I will include the core existing logic but refactored to use the helper.
        
        # A. High Budget (>= 5000円)
        if budget >= 5000:
            # 3連単4頭BOX
            if len(top_horses) >= 4:
                rec = BettingAllocator._create_box_rec('3連単', top_horses[:4], 2400)
                recommendations.append(rec)
                remaining_budget -= 2400
                
            # 3連複5頭BOX
            if len(top_horses) >= 5 and remaining_budget >= 1000:
                rec = BettingAllocator._create_box_rec('3連複', top_horses[:5], 1000)
                recommendations.append(rec)
                remaining_budget -= 1000
                
            # 馬連5頭BOX
            if len(top_horses) >= 5 and remaining_budget >= 1000:
                rec = BettingAllocator._create_box_rec('馬連', top_horses[:5], 1000)
                recommendations.append(rec)
                remaining_budget -= 1000
                
        # B. Mid Budget (>= 2400円)
        elif budget >= 2400:
            if len(top_horses) >= 4:
                rec = BettingAllocator._create_box_rec('3連単', top_horses[:4], 2400)
                recommendations.append(rec)
                remaining_budget -= 2400
                
        # C. Low Budget
        else:
            if budget >= 1000 and len(top_horses) >= 5:
                rec = BettingAllocator._create_box_rec('3連複', top_horses[:5], 1000)
                recommendations.append(rec)
                remaining_budget -= 1000
                
            if remaining_budget >= 1000 and len(top_horses) >= 5:
                rec = BettingAllocator._create_box_rec('ワイド', top_horses[:5], 1000)
                recommendations.append(rec)
                remaining_budget -= 1000
        
        # D. 余剰 (単勝)
        if remaining_budget >= 100:
            win_amount = (remaining_budget // 100) * 100
            rec = {
                'type': '単勝',
                'method': 'SINGLE',
                'formation': [[top_horses[0]]],
                'amount': win_amount,
                'count': 1,
                'desc': '一点買い',
                'horses': [top_horses[0]]
            }
            recommendations.append(rec)

        return BettingAllocator._format_recommendations(recommendations)

    @staticmethod
    def _create_box_rec(bet_type, horses, amount):
        import math
        n = len(horses)
        p = 0
        if bet_type in ['馬連', 'ワイド', '枠連']: p = n * (n - 1) // 2
        elif bet_type == '馬単': p = n * (n - 1)
        elif bet_type == '3連複': p = n * (n - 1) * (n - 2) // 6
        elif bet_type == '3連単': p = n * (n - 1) * (n - 2)
        
        return {
            'type': bet_type,
            'method': 'BOX',
            'formation': [horses],
            'amount': amount,
            'count': p,
            'desc': f'{n}頭BOX',
            'horses': horses
        }

    @staticmethod
    def _format_recommendations(recommendations):
        final_list = []
        for r in recommendations:
            combo_str = "-".join(map(str, r['horses']))
            final_list.append({
                'bet_type': r['type'],
                'method': r['method'],
                'combination': combo_str,
                'description': r['desc'],
                'points': r['count'],
                'unit_amount': 100,
                'total_amount': r['amount'],
                'horse_numbers': r['horses'],
                'reason': '予算最適化'
            })
        return final_list
