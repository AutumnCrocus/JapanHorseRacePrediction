"""
予算配分・買い目生成モジュール
予算に応じて最適な券種ポートフォリオ（3連単BOX、3連複BOX、単勝など）を構築する。
"""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from .strategy_composite import CompositeBettingStrategy
from .strategy import BettingStrategy

class BettingAllocator:
    """
    budget配分を行うクラス
    """
    
    @staticmethod
    def allocate_budget(df_preds, budget: int, odds_data: dict = None, allowed_types: list = None, strategy: str = 'balance') -> list:
        """
        予算に応じて推奨買い目を生成する
        
        Args:
            df_preds: 予測結果DataFrame (probability, horse_number, etc.)
            budget: 予算総額 (円)
            odds_data: オッズデータ (Optional)
            allowed_types: 許可する券種リスト (例: ['単勝'], ['3連単'])。Noneの場合は予算に応じた最適ミックス。
            strategy: 戦略 ('balance', 'formation', 'box')
            
        Returns:
            list: 推奨買い目のリスト
        """
        if strategy == 'formation':
            return BettingAllocator._allocate_formation(df_preds, budget)
            
        recommendations = []
        
        # 予測確率順にソート（この順序でBOX候補を選ぶ）
        df_sorted = df_preds.sort_values('probability', ascending=False).copy()
        top_horses = df_sorted['horse_number'].tolist()
        
        if not top_horses:
            print("DEBUG: No horses found.")
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
            return BettingAllocator._format_recommendations(recommendations, df_preds, odds_data)


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

        return BettingAllocator._format_recommendations(recommendations, df_preds, odds_data)

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
    def _format_recommendations(recommendations, df_preds, odds_data=None):
        final_list = []
        
        type_map = {
            '単勝': 'tan', '複勝': 'fuku',
            '馬連': 'umaren', 'ワイド': 'wide',
            '馬単': 'umatan', '枠連': 'wakuren',
            '3連複': 'sanrenpuku', '3連単': 'sanrentan'
        }
        
        for r in recommendations:
            combo_str = "-".join(map(str, r['horses']))
            
            # 理由生成
            reason = '予算最適化'
            try:
                # 軸馬（リストの先頭）の特徴を取得
                if r['horses']:
                    head_horse = int(r['horses'][0])
                    # df_predsから馬情報を検索
                    # df_predsは辞書リストではなくDataFrame
                    target_row = df_preds[df_preds['horse_number'] == head_horse]
                    
                    if not target_row.empty:
                        row = target_row.iloc[0]
                        feature_dict = row.to_dict()
                        
                        bet_code = type_map.get(r['type'], 'tan')
                        
                        # BettingStrategyを使って理由を生成
                        if r['method'] == 'BOX':
                            reason = BettingStrategy.generate_box_reason(
                                r['type'],
                                r['horses'],
                                df_preds,
                                odds_data
                            )
                        else:
                            reason = BettingStrategy.generate_reason(
                                bet_code,
                                [str(h) for h in r['horses']],
                                row.get('probability', 0.1),
                                row.get('expected_value', 1.0),
                                row.get('odds', 0.0),
                                [feature_dict]
                            )
                        # Add stats for display
                        # IF SINGLE: use head horse stats
                        # IF BOX/Other: Set stats to None (as single odds/ev are misleading) and list all names
                        
                        is_single = (r['method'] == 'SINGLE')
                        
                        # Horse Name Logic
                        horse_names = []
                        for h_num in r['horses']:
                            h_row = df_preds[df_preds['horse_number'] == int(h_num)]
                            if not h_row.empty:
                                name = h_row.iloc[0].get('horse_name') or h_row.iloc[0].get('馬名') or str(h_num)
                                horse_names.append(name)
                            else:
                                horse_names.append(str(h_num))
                        
                        display_name = ",".join(horse_names) if len(horse_names) <= 3 else f"{','.join(horse_names[:3])}..."
                        
                        stats = {
                            'horse_name': display_name,
                            'odds': (row.get('odds', 0) if row.get('odds', 0) > 0 else None) if is_single else None,
                            'prob': row.get('probability') if is_single else None,
                            'ev': row.get('expected_value') if is_single else None
                        }

            except Exception as e:
                print(f"Reason generation error: {e}")
                reason = "予算最適化（詳細生成エラー）"
                stats = {
                    'horse_name': "-",
                    'odds': None,
                    'prob': None,
                    'ev': None
                }

            rec_dict = {
                'bet_type': r['type'],
                'method': r['method'],
                'combination': combo_str,
                'description': r['desc'],
                'points': r['count'],
                'unit_amount': 100,
                'total_amount': r['amount'],
                'horse_numbers': r['horses'],
                'reason': reason
            }
            # Merge stats
            rec_dict.update(stats)
            
            final_list.append(rec_dict)
        return final_list

    @staticmethod
    def _allocate_formation(df_preds, budget):
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top = df_sorted['horse_number'].tolist()
        probs = df_sorted['probability'].tolist()
        
        if len(top) < 6:
            # Fallback
            return BettingAllocator.allocate_budget(df_preds, budget, strategy='balance')
            
        # 1. Analyze Distribution
        # Check if top horses are dominant or flat
        is_flat = (probs[0] - probs[4]) < 0.15 # Top 5 within 15% -> Confused
        is_strong_axis = probs[0] > 0.30 # Top 1 is strong (>30%)
        
        # 2. Strategy Selection & Fallback Loop
        # Priority: 3-Ren-Tan Formation -> 3-Ren-Puku Box/Form -> Uma-Ren
        
        recommendations = []
        remaining_budget = budget
        
        # --- ① 単勝保険ロジック ---
        # 確率上位5位以内かつ期待値10以上の馬を抽出し、オッズで傾斜配分
        insurance_candidates = []
        for i in range(min(5, len(df_sorted))):
            horse_odds = df_sorted.iloc[i]['odds']
            horse_prob = probs[i]
            horse_ev = horse_prob * horse_odds
            horse_num = top[i]
            
            if horse_ev >= 10.0:
                insurance_candidates.append({
                    'horse_num': horse_num,
                    'odds': horse_odds,
                    'prob': horse_prob,
                    'ev': horse_ev,
                    'rank': i + 1
                })
        
        # 保険対象馬が存在する場合、予算の一部を単勝に配分
        if insurance_candidates and remaining_budget >= 300:
            # 単勝保険に使う予算は全体の最大30%、または1頭あたり最大500円
            max_insurance_budget = min(int(budget * 0.3), 500 * len(insurance_candidates))
            insurance_budget = min(max_insurance_budget, remaining_budget - 300)  # 最低300円はメインベットに残す
            
            if insurance_budget >= 100:
                # オッズ合計で傾斜配分 (高オッズほど多く配分)
                total_odds = sum(c['odds'] for c in insurance_candidates)
                
                for cand in insurance_candidates:
                    weight = cand['odds'] / total_odds
                    amt = int(insurance_budget * weight / 100) * 100
                    if amt < 100: amt = 100
                    if amt > remaining_budget: amt = remaining_budget
                    
                    if amt >= 100:
                        recommendations.append({
                            'bet_type': '単勝',
                            'method': 'SINGLE',
                            'combination': str(cand['horse_num']),
                            'horse_numbers': [cand['horse_num']],
                            'total_amount': amt,
                            'points': 1,
                            'description': f'高期待値単勝({cand["rank"]}位)',
                            'reason': f'期待値{cand["ev"]:.1f} (確率{int(cand["prob"]*100)}%×オッズ{cand["odds"]:.1f})'
                        })
                        remaining_budget -= amt

        # --- Main Bet Logic ---
        
        def calculate_cost(bet_type, method, g1, g2, g3=None):
            pts = 0
            if bet_type == '3連単':
                # Formation: g1 -> g2 -> g3
                for i in g1:
                    for j in g2:
                        if i==j: continue
                        for k in g3:
                            if k==i or k==j: continue
                            pts+=1
            elif bet_type == '3連複':
                 if method == 'BOX':
                    n = len(g1)
                    pts = n * (n-1) * (n-2) // 6
                 else: # Formation 1-Axis: g1 -> g2 (actually g2 is list of opponents)
                    # 3-Ren-Puku 1-Head Axis: g1[0] - g2 - g2
                    head = g1[0]
                    opps = [x for x in g2 if x != head]
                    n = len(opps)
                    pts = n * (n-1) // 2
            elif bet_type == '馬連':
                if method == 'BOX':
                    n = len(g1)
                    pts = n * (n-1) // 2
                else: # Formation/Nagashi
                     head = g1[0]
                     opps = [x for x in g2 if x != head]
                     pts = len(opps)
            elif bet_type == 'ワイド':
                 # Box
                 n = len(g1)
                 pts = n * (n-1) // 2
                 
            return pts * 100
            
        main_bet = None
        
        # Try 3-Ren-Tan Formation (High Budget, Hierarchy)
        # Conditions: Not Flat OR Budget sufficient
        if not is_flat and remaining_budget >= 1000:
            # 1st: Top 2, 2nd: Top 5, 3rd: Top 6 (Wide Formation)
            g1, g2, g3 = top[:2], top[:5], top[:6]
            cost = calculate_cost('3連単', 'FORMATION', g1, g2, g3)
            
            if cost > remaining_budget:
                # Shrink: 1st: Top 1, 2nd: Top 4, 3rd: Top 5
                g1, g2, g3 = top[:1], top[:4], top[:5]
                cost = calculate_cost('3連単', 'FORMATION', g1, g2, g3)
            
            if cost <= remaining_budget:
                main_bet = {
                    'type': '3連単', 'method': 'FORMATION', 'g1': g1, 'g2': g2, 'g3': g3, 'cost': cost, 'pts': cost//100
                }

        # Try 3-Ren-Puku (If 3-Ren-Tan failed or Flat)
        if not main_bet:
            if is_flat:
                 # Box 5
                 g1 = top[:5]
                 cost = calculate_cost('3連複', 'BOX', g1, None)
                 if cost <= remaining_budget:
                     main_bet = {'type': '3連複', 'method': 'BOX', 'g1': g1, 'cost': cost, 'pts': cost//100}
            else:
                 # 1-Head Axis 6 Flow
                 g1, g2 = top[:1], top[:6]
                 cost = calculate_cost('3連複', 'FORMATION', g1, g2)
                 if cost <= remaining_budget:
                     main_bet = {'type': '3連複', 'method': '流し', 'g1': g1, 'g2': g2, 'cost': cost, 'pts': cost//100}

        # Try Uma-Ren (Low Budget Fallback)
        if not main_bet:
             if is_flat:
                 # Box 5
                 g1 = top[:5]
                 cost = calculate_cost('馬連', 'BOX', g1, None)
                 if cost <= remaining_budget:
                     main_bet = {'type': '馬連', 'method': 'BOX', 'g1': g1, 'cost': cost, 'pts': cost//100}
             else:
                 # Nagashi 5
                 g1, g2 = top[:1], top[:6]
                 cost = calculate_cost('馬連', '流し', g1, g2)
                 if cost <= remaining_budget:
                    main_bet = {'type': '馬連', 'method': '流し', 'g1': g1, 'g2': g2, 'cost': cost, 'pts': cost//100}
                    
        # Try Wide (Ultimate Fallback)
        if not main_bet:
             g1 = top[:4]
             cost = calculate_cost('ワイド', 'BOX', g1, None)
             if cost <= remaining_budget:
                  main_bet = {'type': 'ワイド', 'method': 'BOX', 'g1': g1, 'cost': cost, 'pts': cost//100}
                  
        # Apply Main Bet
        if main_bet:
            # Scale amount
            unit = (remaining_budget // main_bet['cost']) * 100
            if unit < 100: unit = 100 # Should not happen if cost <= budget check passed
            
            rec = {
                'bet_type': main_bet['type'],
                'method': main_bet['method'],
                'points': main_bet['pts'],
                'unit_amount': unit,
                'total_amount': main_bet['pts'] * unit,
                'reason': '混戦BOX' if is_flat and 'BOX' in main_bet['method'] else '軸強固・フォーメーション',
                'horse_numbers': main_bet['g1'] + (main_bet.get('g2') or []) + (main_bet.get('g3') or [])
            }
            
            # Format combination string
            if main_bet['method'] == 'BOX':
                 rec['combination'] = f"{main_bet['g1']} BOX"
                 rec['formation'] = [main_bet['g1']]
            elif main_bet['type'] == '3連単':
                 rec['combination'] = f"1着:{main_bet['g1']} - 2着:{main_bet['g2']} - 3着:{main_bet['g3']}"
                 rec['formation'] = [main_bet['g1'], main_bet['g2'], main_bet['g3']]
            else:
                 rec['combination'] = f"軸:{main_bet['g1']} - 相手:{main_bet['g2']}"
                 rec['formation'] = [main_bet['g1'], main_bet['g2']]

            rec['horse_numbers'] = list(set(rec['horse_numbers'])) # Unique
            recommendations.append(rec)
            
        return recommendations
