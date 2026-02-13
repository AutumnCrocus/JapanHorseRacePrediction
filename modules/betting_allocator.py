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
    def allocate_budget(df_preds: pd.DataFrame, budget: int, odds_data: dict = None, allowed_types: list = None, strategy: str = 'balance', bias_info: dict = None) -> list:
        """
        予算配分を実行するメインメソッド
        
        Args:
            df_preds: 予測結果DataFrame
            budget: 予算総額 (円)
            odds_data: オッズデータ (Optional)
            allowed_types: 許可する券種リスト (例: ['単勝'], ['3連単'])。Noneの場合は予算に応じた最適ミックス。
            strategy: 戦略 ('balance', 'formation', 'box', 'kelly', 'odds_divergence', 'track_bias')
            bias_info: トラックバイアス情報 (track_bias戦略用)
            
        Returns:
            list: 推奨買い目のリスト
        """
        recommendations = []
        
        if strategy == 'formation':
            recommendations = BettingAllocator._allocate_formation(df_preds, budget)
        elif strategy == 'hybrid_1000':
            recommendations = BettingAllocator._allocate_hybrid_1000(df_preds, budget)
        elif strategy == 'kelly':
            recommendations = BettingAllocator._allocate_kelly(df_preds, budget, odds_data)
        elif strategy == 'odds_divergence':
            recommendations = BettingAllocator._allocate_odds_divergence(df_preds, budget, odds_data)
        elif strategy == 'track_bias':
            recommendations = BettingAllocator._allocate_track_bias(df_preds, budget, bias_info)
        elif strategy == 'formation_flex':
            recommendations = BettingAllocator._allocate_formation_flex(df_preds, budget)
        elif strategy == 'wide_nagashi':
            recommendations = BettingAllocator._allocate_wide_nagashi(df_preds, budget)
        elif strategy == 'box4_umaren':
            recommendations = BettingAllocator._allocate_box4_umaren(df_preds, budget)
        elif strategy == 'box4_sanrenpuku':
            recommendations = BettingAllocator._allocate_box4_sanrenpuku(df_preds, budget)
        elif strategy == 'box5_sanrenpuku':
            recommendations = BettingAllocator._allocate_box5_sanrenpuku(df_preds, budget)
        elif strategy == 'umaren_nagashi':
            recommendations = BettingAllocator._allocate_umaren_nagashi(df_preds, budget)
        elif strategy == 'sanrenpuku_1axis':
            recommendations = BettingAllocator._allocate_sanrenpuku_1axis(df_preds, budget)
        elif strategy == 'sanrenpuku_2axis':
            recommendations = BettingAllocator._allocate_sanrenpuku_2axis(df_preds, budget)
        elif strategy == 'meta_optimized':
            recommendations = BettingAllocator._allocate_meta_optimized(df_preds, budget)
        elif strategy == 'meta_contrarian':
            recommendations = BettingAllocator._allocate_meta_contrarian(df_preds, budget)
        elif strategy == 'ranking_anchor':
            recommendations = BettingAllocator._allocate_ranking_anchor(df_preds, budget)
            
        if recommendations:
            return BettingAllocator._format_recommendations(recommendations, df_preds, odds_data)
        
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
            # 買い目文字列の生成 (methodに基づいて動的に生成)
            method = r.get('method', 'SINGLE')
            formation = r.get('formation', [])
            horses = r.get('horses', r.get('horse_numbers', []))
            
            if r.get('combination'):
                # すでに生成済みの場合はそれを使用（ただし馬番のみの場合は整形）
                combo_str = r['combination']
            elif method == 'BOX':
                combo_str = f"{','.join(map(str, horses))} BOX"
            elif method in ['流し', 'NAGASHI', '1軸流し']:
                if len(formation) >= 2:
                    combo_str = f"軸:{','.join(map(str, formation[0]))} - 相手:{','.join(map(str, formation[1]))}"
                else:
                    combo_str = f"軸:{horses[0]} - 相手:{','.join(map(str, horses[1:]))}"
            elif method == 'FORMATION':
                if len(formation) == 3:
                    combo_str = f"1着:{formation[0]} 2着:{formation[1]} 3着:{formation[2]}"
                elif len(formation) == 2:
                    combo_str = f"軸:{formation[0]} 相手:{formation[1]}"
                else:
                    combo_str = "-".join(map(str, horses))
            else:
                combo_str = "-".join(map(str, horses))
            
            # 理由生成
            reason = '予算最適化'
            try:
                # 軸馬（リストの先頭）の特徴を取得
                if horses:
                    head_horse = int(horses[0])
                    # df_predsから馬情報を検索
                    # df_predsは辞書リストではなくDataFrame
                    target_row = df_preds[df_preds['horse_number'] == head_horse]
                    
                    if not target_row.empty:
                        row = target_row.iloc[0]
                        feature_dict = row.to_dict()
                        
                        bet_code = type_map.get(r['type'], 'tan')
                        
                        # Reasoning (SHAP) を取得
                        reasoning = row.get('reasoning', {})
                        
                        # BettingStrategyを使って理由を生成
                        if method == 'BOX':
                            reason = BettingStrategy.generate_box_reason(
                                r.get('type', 'BOX'),
                                horses,
                                df_preds,
                                odds_data
                            )
                        else:
                            reason = BettingStrategy.generate_reason(
                                bet_code,
                                [str(h) for h in horses],
                                row.get('probability', 0.1),
                                row.get('expected_value', 1.0),
                                row.get('odds', 0.0),
                                [feature_dict],
                                reasoning=reasoning
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
                # Silently handle reason generation errors (too noisy)
                reason = "予算最適化（詳細生成エラー）"
                stats = {
                    'horse_name': "-",
                    'odds': None,
                    'prob': None,
                    'ev': None
                }

            rec_dict = {
                'bet_type': r.get('type', r.get('bet_type')),
                'method': method,
                'combination': combo_str,
                'description': r.get('desc', r.get('description', '')),
                'points': r.get('count', r.get('points', 1)),
                'unit_amount': r.get('unit_amount', 100),
                'total_amount': r.get('amount', r.get('total_amount', 0)),
                'horse_numbers': horses,
                'reason': reason,
                'formation': formation
            }
            # Merge stats
            rec_dict.update(stats)
            
            final_list.append(rec_dict)
        return final_list

    @staticmethod
    def _allocate_hybrid_1000(df_preds, budget):
        """
        予算1000円専用: 超厳選Formation + Balance複合戦略
        - 軸が強固（混戦でない）場合: 3連複軸1頭流し（相手5頭=10点=1000円）
        - 混戦の場合: ワイド4頭BOX（6点=600円）+ 単勝配分（400円）
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top = df_sorted['horse_number'].tolist()
        probs = df_sorted['probability'].tolist()
        
        if not top: return []
        
        # 混戦判定 (1位と5位の確率差が小さい、または1位の信頼度が低い)
        is_flat = False
        if len(probs) >= 5:
            is_flat = (probs[0] - probs[4]) < 0.15
        if probs[0] < 0.20: # 1位の確率が20%未満なら絶対混戦
            is_flat = True
            
        recommendations = []
        
        # 1. Strict Formation (Strong Axis)
        # 条件: 予算1000円以上、かつ混戦でない、かつ相手が5頭以上いる
        if budget >= 1000 and (not is_flat) and len(top) >= 6:
            axis = top[0]
            opponents = top[1:6] # 2位〜6位 (5頭)
            
            # 3連複 軸1頭流し (相手5頭) = 5C2 = 10点 = 1000円
            rec = {
                'type': '3連複',
                'method': '流し',
                'desc': '厳選3連複(軸1頭流し)',
                'horses': [axis] + opponents,
                'formation': [[axis], opponents],
                'count': 10,
                'amount': 1000
            }
            recommendations.append(rec)
            return recommendations

        # 2. Balance Strategy (Fallback / Flat)
        # ワイドBOX + 単勝
        remaining_budget = budget
        
        # ワイド4頭BOX (6点=600円)
        if remaining_budget >= 600 and len(top) >= 4:
            box_horses = top[:4]
            rec = {
                'type': 'ワイド',
                'method': 'BOX',
                'desc': 'ワイド混戦BOX',
                'horses': box_horses,
                'count': 6,
                'amount': 600
            }
            recommendations.append(rec)
            remaining_budget -= 600
            
        # 残りで単勝 (上位馬に配分)
        if remaining_budget >= 100 and top:
            # 1位〜3位に配分
            targets = top[:3]
            idx = 0
            while remaining_budget >= 100:
                h = targets[idx % len(targets)]
                
                # 既存リストから探すか新規作成
                found = False
                for r in recommendations:
                    if r.get('type') == '単勝' and r.get('horses', [None])[0] == h:
                        r['amount'] = r.get('amount', 0) + 100
                        found = True
                        break
                
                if not found:
                    recommendations.append({
                        'type': '単勝',
                        'method': 'SINGLE',
                        'desc': '単勝プッシュ',
                        'horses': [h],
                        'count': 1,
                        'amount': 100
                    })
                
                remaining_budget -= 100
                idx += 1
                
        return recommendations


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
                    'rank': i + 1,
                    'formation': None # Placeholder, as single bets don't have a formation in this context
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
                'type': main_bet['type'],
                'method': main_bet['method'],
                'count': main_bet['pts'],
                'amount': main_bet['pts'] * unit,
                'horses': main_bet['g1'] + (main_bet.get('g2') or []) + (main_bet.get('g3') or []),
                'formation': [main_bet['g1']] + ([main_bet['g2']] if 'g2' in main_bet else []) + ([main_bet['g3']] if 'g3' in main_bet else []),
                'desc': '混戦BOX' if is_flat and 'BOX' in main_bet['method'] else '軸強固・フォーメーション'
            }
            recommendations.append(rec)
            
        return recommendations

    @staticmethod
    def _allocate_kelly(df_preds: pd.DataFrame, budget: int, odds_data: dict = None,
                       use_half_kelly: bool = False, min_edge: float = 0.0) -> list:
        """
        Kelly基準に基づく予算配分（馬連・三連複対応版）
        
        戦略: 
        - 上位馬の単勝、馬連、3連複を期待値で比較
        - 最も期待値が高い券種・組み合わせを選択
        
        Args:
            df_preds: 予測DataFrame
            budget: 予算
            odds_data: オッズデータ
            
        Returns:
            推奨リスト
        """
        from itertools import combinations
        
        recommendations = []
        
        if df_preds.empty or budget <= 0:
            return recommendations
        
        # 予測確率でソート、上位5頭に限定（計算量削減）
        df_sorted = df_preds.sort_values('probability', ascending=False).head(5).copy()
        
        # 馬情報を抽出
        horses = []
        for _, row in df_sorted.iterrows():
            prob = row.get('probability', 0)
            horse_num = int(row.get('horse_number', 0))
            
            if prob <= 0 or horse_num == 0:
                continue
            
            # 単勝オッズ取得
            odds = row.get('odds', 0)
            if odds <= 1 and odds_data:
                tan_odds = odds_data.get('tan', {})
                if horse_num in tan_odds:
                    odds = tan_odds[horse_num]
            
            horses.append({
                'num': horse_num,
                'name': row.get('horse_name', row.get('name', f'馬番{horse_num}')),
                'prob': prob,
                'odds': odds if odds > 1 else 2.0
            })
        
        if len(horses) < 2:
            # 馬が足りない場合は単勝のみ
            if horses:
                h = horses[0]
                amount = (budget // 100) * 100
                ev = h['prob'] * h['odds']
                recommendations.append({
                    'bet_type': '単勝', 'type': '単勝', 'method': 'SINGLE',
                    'horse_numbers': [h['num']], 'formation': [[h['num']]],
                    'combination': str(h['num']), 'unit_amount': amount,
                    'total_amount': amount, 'pts': 1, 'odds': h['odds'],
                    'prob': h['prob'], 'ev': ev,
                    'reason': f"単勝集中: 期待値{ev:.2f}倍",
                    'horse_name': h['name']
                })
            return recommendations
        
        # ========== 全候補の期待値を計算 ==========
        all_bets = []
        
        # 1. 単勝候補（上位3頭）
        for h in horses[:3]:
            ev = h['prob'] * h['odds']
            if ev > 0.8:  # 期待値0.8以上
                all_bets.append({
                    'bet_type': '単勝', 'horses': [h],
                    'odds': h['odds'], 'prob': h['prob'], 'ev': ev,
                    'combo_str': str(h['num']),
                    'horse_nums': [h['num']]
                })
        
        # 2. 馬連候補（上位馬の2頭組み合わせ）
        if odds_data and 'umaren' in odds_data:
            umaren_odds = odds_data['umaren']
            for h1, h2 in combinations(horses[:4], 2):
                key = tuple(sorted([h1['num'], h2['num']]))
                if key in umaren_odds:
                    odds = umaren_odds[key]
                    # 馬連確率: 両馬が上位2着以内に入る確率（簡易推定）
                    # P(両方上位2) ≒ P(h1) × P(h2) × 調整係数
                    combo_prob = h1['prob'] * h2['prob'] * 1.5  # 調整係数
                    combo_prob = min(combo_prob, 0.5)  # 上限50%
                    ev = combo_prob * odds
                    if ev > 1.0:
                        all_bets.append({
                            'bet_type': '馬連', 'horses': [h1, h2],
                            'odds': odds, 'prob': combo_prob, 'ev': ev,
                            'combo_str': f"{h1['num']}-{h2['num']}",
                            'horse_nums': [h1['num'], h2['num']]
                        })
        
        # 3. 三連複候補（上位馬の3頭組み合わせ）
        if odds_data and 'sanrenpuku' in odds_data and len(horses) >= 3:
            sanrenpuku_odds = odds_data['sanrenpuku']
            for h1, h2, h3 in combinations(horses[:5], 3):
                key = tuple(sorted([h1['num'], h2['num'], h3['num']]))
                if key in sanrenpuku_odds:
                    odds = sanrenpuku_odds[key]
                    # 三連複確率: 3頭が上位3着以内に入る確率（簡易推定）
                    combo_prob = h1['prob'] * h2['prob'] * h3['prob'] * 3.0
                    combo_prob = min(combo_prob, 0.3)  # 上限30%
                    ev = combo_prob * odds
                    if ev > 1.2:  # 期待値1.2以上
                        all_bets.append({
                            'bet_type': '三連複', 'horses': [h1, h2, h3],
                            'odds': odds, 'prob': combo_prob, 'ev': ev,
                            'combo_str': f"{h1['num']}-{h2['num']}-{h3['num']}",
                            'horse_nums': [h1['num'], h2['num'], h3['num']]
                        })
        
        # ========== 期待値でソート ==========
        all_bets.sort(key=lambda x: x['ev'], reverse=True)
        
        if not all_bets:
            # 候補がない場合、上位1頭に全額単勝
            h = horses[0]
            amount = (budget // 100) * 100
            ev = h['prob'] * h['odds']
            recommendations.append({
                'bet_type': '単勝', 'type': '単勝', 'method': 'SINGLE',
                'horse_numbers': [h['num']], 'formation': [[h['num']]],
                'combination': str(h['num']), 'unit_amount': amount,
                'total_amount': amount, 'pts': 1, 'odds': h['odds'],
                'prob': h['prob'], 'ev': ev,
                'reason': f"予測1位集中: 的中時{int(amount * h['odds'])}円",
                'horse_name': h['name']
            })
            return recommendations
        
        # ========== 予算配分 ==========
        # 最高EV候補に60%、2番目に30%、3番目に10%
        top_bets = all_bets[:3]
        alloc_ratios = [0.6, 0.3, 0.1] if len(top_bets) >= 3 else \
                       [0.7, 0.3] if len(top_bets) == 2 else [1.0]
        
        for i, bet in enumerate(top_bets):
            if i >= len(alloc_ratios):
                break
            
            alloc = alloc_ratios[i]
            amount = int(budget * alloc / 100) * 100
            
            if amount < 100:
                continue
            
            profit = int(amount * bet['odds']) - amount
            horse_names = ', '.join([h['name'] for h in bet['horses']])
            
            # formation形式を生成
            if bet['bet_type'] == '単勝':
                formation = [[bet['horse_nums'][0]]]
                method = 'SINGLE'
            elif bet['bet_type'] == '馬連':
                formation = [bet['horse_nums']]
                method = 'BOX'
            else:  # 三連複
                formation = [bet['horse_nums']]
                method = 'BOX'
            
            recommendations.append({
                'bet_type': bet['bet_type'],
                'type': bet['bet_type'],
                'method': method,
                'horse_numbers': bet['horse_nums'],
                'formation': formation,
                'combination': bet['combo_str'],
                'unit_amount': amount,
                'total_amount': amount,
                'pts': 1,
                'odds': bet['odds'],
                'prob': bet['prob'],
                'ev': bet['ev'],
                'reason': f"期待値{bet['ev']:.2f}倍, 的中時+{profit}円",
                'horse_name': horse_names
            })
        
        return recommendations

    @staticmethod
    def _allocate_odds_divergence(df_preds: pd.DataFrame, budget: int, odds_data: dict = None) -> list:
        """
        オッズ乖離戦略に基づく予算配分
        
        戦略:
        - AIの予測確率 vs 市場オッズの暗示確率を比較
        - 乖離が大きい（過小評価されている）馬を狙う
        - 乖離率 = (予測確率 - 暗示確率) / 暗示確率
        
        Args:
            df_preds: 予測DataFrame
            budget: 予算
            odds_data: オッズデータ
            
        Returns:
            推奨リスト
        """
        from itertools import combinations
        
        recommendations = []
        
        if df_preds.empty or budget <= 0:
            return recommendations
        
        # 予測確率でソート
        df_sorted = df_preds.sort_values('probability', ascending=False).copy()
        
        # 各馬のオッズ乖離を計算
        divergence_data = []
        for _, row in df_sorted.iterrows():
            model_prob = row.get('probability', 0)
            horse_num = int(row.get('horse_number', 0))
            
            if model_prob <= 0 or horse_num == 0:
                continue
            
            # 単勝オッズ取得
            tan_odds = row.get('odds', 0)
            if tan_odds <= 1 and odds_data:
                tan_odds_dict = odds_data.get('tan', {})
                if horse_num in tan_odds_dict:
                    tan_odds = tan_odds_dict[horse_num]
            
            if tan_odds <= 1:
                tan_odds = 2.0  # デフォルト
            
            # 市場暗示確率 = 1 / オッズ（控除率考慮で0.8掛け）
            market_implied_prob = 0.8 / tan_odds
            
            # 乖離率 = (予測 - 暗示) / 暗示
            # 正の値 = AIの方が高く評価 = 市場が過小評価
            divergence = (model_prob - market_implied_prob) / market_implied_prob if market_implied_prob > 0 else 0
            
            # 期待値
            ev = model_prob * tan_odds
            
            divergence_data.append({
                'num': horse_num,
                'name': row.get('horse_name', row.get('name', f'馬番{horse_num}')),
                'model_prob': model_prob,
                'market_prob': market_implied_prob,
                'divergence': divergence,
                'odds': tan_odds,
                'ev': ev
            })
        
        if not divergence_data:
            return recommendations
        
        # === オッズ乖離でソート（大きい順 = 過小評価順）===
        divergence_data.sort(key=lambda x: x['divergence'], reverse=True)
        
        # === 全券種の期待値を計算 ===
        all_bets = []
        
        # 1. 単勝候補（乖離上位3頭）
        for h in divergence_data[:3]:
            if h['divergence'] > 0.1:  # 10%以上の乖離
                all_bets.append({
                    'bet_type': '単勝',
                    'horses': [h],
                    'odds': h['odds'],
                    'prob': h['model_prob'],
                    'ev': h['ev'],
                    'divergence': h['divergence'],
                    'combo_str': str(h['num']),
                    'horse_nums': [h['num']]
                })
        
        # 2. 馬連候補（乖離上位4頭から組み合わせ）
        top4 = divergence_data[:4]
        if odds_data and 'umaren' in odds_data and len(top4) >= 2:
            umaren_odds = odds_data['umaren']
            for h1, h2 in combinations(top4, 2):
                key = tuple(sorted([h1['num'], h2['num']]))
                if key in umaren_odds:
                    odds = umaren_odds[key]
                    # 組み合わせの乖離 = 平均乖離
                    avg_divergence = (h1['divergence'] + h2['divergence']) / 2
                    combo_prob = h1['model_prob'] * h2['model_prob'] * 1.5
                    combo_prob = min(combo_prob, 0.5)
                    ev = combo_prob * odds
                    
                    if avg_divergence > 0.05 and ev > 1.0:  # 5%以上の乖離
                        all_bets.append({
                            'bet_type': '馬連',
                            'horses': [h1, h2],
                            'odds': odds,
                            'prob': combo_prob,
                            'ev': ev,
                            'divergence': avg_divergence,
                            'combo_str': f"{h1['num']}-{h2['num']}",
                            'horse_nums': [h1['num'], h2['num']]
                        })
        
        # 3. 三連複候補（乖離上位5頭から組み合わせ）
        top5 = divergence_data[:5]
        if odds_data and 'sanrenpuku' in odds_data and len(top5) >= 3:
            sanrenpuku_odds = odds_data['sanrenpuku']
            for h1, h2, h3 in combinations(top5, 3):
                key = tuple(sorted([h1['num'], h2['num'], h3['num']]))
                if key in sanrenpuku_odds:
                    odds = sanrenpuku_odds[key]
                    avg_divergence = (h1['divergence'] + h2['divergence'] + h3['divergence']) / 3
                    combo_prob = h1['model_prob'] * h2['model_prob'] * h3['model_prob'] * 3.0
                    combo_prob = min(combo_prob, 0.3)
                    ev = combo_prob * odds
                    
                    if avg_divergence > 0 and ev > 1.2:
                        all_bets.append({
                            'bet_type': '3連複',
                            'horses': [h1, h2, h3],
                            'odds': odds,
                            'prob': combo_prob,
                            'ev': ev,
                            'divergence': avg_divergence,
                            'combo_str': f"{h1['num']}-{h2['num']}-{h3['num']}",
                            'horse_nums': [h1['num'], h2['num'], h3['num']]
                        })
        
        # === 乖離×期待値でスコア計算してソート ===
        # スコア = 乖離率 × 期待値（両方を考慮）
        for bet in all_bets:
            bet['score'] = bet['divergence'] * bet['ev']
        
        all_bets.sort(key=lambda x: x['score'], reverse=True)
        
        if not all_bets:
            # 候補がない場合、乖離1位に全額単勝
            h = divergence_data[0]
            amount = (budget // 100) * 100
            recommendations.append({
                'bet_type': '単勝', 'type': '単勝', 'method': 'SINGLE',
                'horse_numbers': [h['num']], 'formation': [[h['num']]],
                'combination': str(h['num']), 'unit_amount': amount,
                'total_amount': amount, 'pts': 1, 'odds': h['odds'],
                'prob': h['model_prob'], 'ev': h['ev'],
                'reason': f"乖離率{h['divergence']*100:.0f}%, AI評価{h['model_prob']*100:.0f}% vs 市場{h['market_prob']*100:.0f}%",
                'horse_name': h['name']
            })
            return recommendations
        
        # === 予算配分 ===
        top_bets = all_bets[:3]
        alloc_ratios = [0.6, 0.3, 0.1] if len(top_bets) >= 3 else \
                       [0.7, 0.3] if len(top_bets) == 2 else [1.0]
        
        for i, bet in enumerate(top_bets):
            if i >= len(alloc_ratios):
                break
            
            alloc = alloc_ratios[i]
            amount = int(budget * alloc / 100) * 100
            
            if amount < 100:
                continue
            
            profit = int(amount * bet['odds']) - amount
            horse_names = ', '.join([h['name'] for h in bet['horses']])
            
            # 乖離率表示（単勝の場合は個別、組み合わせの場合は平均）
            if bet['bet_type'] == '単勝':
                div_pct = bet['horses'][0]['divergence'] * 100
            else:
                div_pct = bet['divergence'] * 100
            
            # formation形式を生成
            if bet['bet_type'] == '単勝':
                formation = [[bet['horse_nums'][0]]]
                method = 'SINGLE'
            else:
                formation = [bet['horse_nums']]
                method = 'BOX'
            
            recommendations.append({
                'bet_type': bet['bet_type'],
                'type': bet['bet_type'],
                'method': method,
                'horse_numbers': bet['horse_nums'],
                'formation': formation,
                'combination': bet['combo_str'],
                'unit_amount': amount,
                'total_amount': amount,
                'pts': 1,
                'odds': bet['odds'],
                'prob': bet['prob'],
                'ev': bet['ev'],
                'reason': f"乖離+{div_pct:.0f}%, 期待値{bet['ev']:.2f}倍, 的中時+{profit}円",
                'horse_name': horse_names
            })
        
        return recommendations

    @staticmethod
    def _allocate_track_bias(df_preds: pd.DataFrame, budget: int, bias_info: dict = None) -> list:
        """
        トラックバイアス特化戦略
        
        Args:
            df_preds: 予測DataFrame
            budget: 予算
            bias_info: 当日のバイアス情報 {'frame_bias': ..., 'position_bias': ...}
            
        Returns:
            list: 推奨リスト
        """
        if df_preds.empty or budget <= 0 or not bias_info:
            return []
            
        frame_bias = bias_info.get('frame_bias', 'flat')
        position_bias = bias_info.get('position_bias', 'flat')
        
        # バイアスなしの場合は見送り
        if frame_bias == 'flat' and position_bias == 'flat':
            return []
            
        # 候補馬の抽出とスコアリング
        candidates = []
        for _, row in df_preds.iterrows():
            score = row['probability']
            h_num = int(row.get('horse_number', 0))
            waku = int(row.get('枠番', 0))
            
            # 1. 枠順バイアス補正
            if frame_bias == 'inner':
                if waku <= 4: score *= 1.2
                elif waku >= 7: score *= 0.8
            elif frame_bias == 'outer':
                if waku >= 6: score *= 1.2
                elif waku <= 3: score *= 0.8
                
            # 2. 脚質バイアス補正 (簡易的)
            # running_styleなどが特徴量にあれば使うが、なければ見送り
            
            candidates.append({
                'num': h_num,
                'name': row.get('horse_name', row.get('name', f'馬番{h_num}')),
                'score': score,
                'prob': row['probability'],
                'odds': row.get('odds', 0)
            })
            
        # スコア順にソート
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 上位馬を軸にする
        axis_horse = candidates[0]
        opponent_horses = candidates[1:6] # 相手5頭
        
        # 軸馬の信頼度が低い場合は見送り
        if axis_horse['score'] < 0.15: # 補正後スコア
            return []
            
        unit_amount = budget // 10
        if unit_amount < 100: unit_amount = 100
        
        recommendations = []
        
        # 馬連流し
        formation_umaren = []
        for opp in opponent_horses:
            formation_umaren.append([axis_horse['num'], opp['num']])
            
        recommendations.append({
            'type': '馬連', 'method': '流し',
            'horses': [axis_horse['num']] + [h['num'] for h in opponent_horses],
            'formation': [[axis_horse['num']], [h['num'] for h in opponent_horses]],
            'amount': unit_amount * len(opponent_horses),
            'count': len(opponent_horses),
            'desc': f"バイアス合致: {frame_bias}/{position_bias}",
            'reason': f"バイアス:{frame_bias}/{position_bias}"
        })
        
        return recommendations

    @staticmethod
    def _allocate_formation_flex(df_preds: pd.DataFrame, budget: int) -> list:
        """
        Formation戦略の改良版 (動的相手選定)
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        all_horses = df_sorted['horse_number'].tolist()
        probs = df_sorted['probability'].tolist()
        
        if len(all_horses) < 6:
            return BettingAllocator.allocate_budget(df_preds, budget, strategy='balance')
            
        axis_horse = all_horses[0]
        axis_prob = probs[0]
        
        # 動的相手選定
        opponents = []
        cum_prob = 0
        candidates = df_sorted.iloc[1:]
        
        for _, row in candidates.iterrows():
            if row['probability'] < 0.02 and len(opponents) >= 5: break
            opponents.append(int(row['horse_number']))
            cum_prob += row['probability']
            if len(opponents) >= 10: break
            
            threshold = 0.80 if axis_prob > 0.3 else 0.90
            # 簡易閾値判定 (累積確率計算は省略、頭数制限と最低確率で制御)
             
        while len(opponents) < 5 and len(opponents) + 1 < len(all_horses):
             opponents.append(all_horses[len(opponents)+1])
             
        # 軸信頼度判定
        bet_type = '3連複'
        combo_count = len(opponents) * (len(opponents) - 1) // 2
        unit_cost = 100
        total_cost = combo_count * unit_cost
        
        if budget >= 2000 and axis_prob > 0.25:
             bet_type = '3連単'
             combo_count = len(opponents) * (len(opponents) - 1)
             total_cost_tan = combo_count * unit_cost
             if total_cost_tan <= budget:
                 total_cost = total_cost_tan
             else:
                 bet_type = '3連複'
        
        recommendations = []
        if total_cost <= budget:
             rec = {
                'bet_type': bet_type,
                'method': '流し' if bet_type == '3連複' else 'FORMATION',
                'type': bet_type,
                'horse_numbers': [axis_horse] + opponents,
                'formation': [[axis_horse], opponents] if bet_type == '3連複' else [[axis_horse], opponents, opponents],
                'points': combo_count,
                'unit_amount': 100,
                'total_amount': total_cost,
                'combination': f"軸:{axis_horse} 相手:{','.join(map(str, opponents))}",
                'reason': f"Flex: 軸信頼度{int(axis_prob*100)}%, 相手{len(opponents)}頭"
             }
             recommendations.append(rec)
             
             remaining = budget - total_cost
             if remaining >= 200:
                 rec_win = {
                     'bet_type': '単勝', 'method': 'SINGLE', 'type': '単勝',
                     'horse_numbers': [axis_horse], 'formation': [[axis_horse]],
                     'points': 1, 'unit_amount': remaining, 'total_amount': remaining,
                     'combination': str(axis_horse), 'reason': '保険'
                 }
                 recommendations.append(rec_win)
        else:
             return BettingAllocator._allocate_formation(df_preds, budget)
             
        return recommendations

    @staticmethod
    def _allocate_balance_flex(df_preds: pd.DataFrame, budget: int) -> list:
        """
        Balance戦略の改良版 (比率配分)
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        top = df_sorted['horse_number'].tolist()
        if len(top) < 5: return BettingAllocator.allocate_budget(df_preds, budget, strategy='balance')
        
        recommendations = []
        b_win = int(budget * 0.20)
        b_wide = int(budget * 0.40)
        b_trip = int(budget * 0.40)
        
        # 1. Win
        if b_win >= 100:
            targets = df_sorted.iloc[:3]
            total_prob = targets['probability'].sum()
            for _, row in targets.iterrows():
                alloc = int((b_win * (row['probability'] / total_prob)) / 100) * 100
                if alloc >= 100:
                    recommendations.append({
                        'bet_type': '単勝', 'type': '単勝', 'method': 'SINGLE',
                        'horse_numbers': [int(row['horse_number'])], 'formation': [[int(row['horse_number'])]],
                        'points': 1, 'unit_amount': alloc, 'total_amount': alloc,
                        'combination': str(int(row['horse_number'])), 'reason': 'Flex Bal: Win'
                    })
                    
        # 2. Wide Box
        if b_wide >= 1000:
            pts = 10
            unit = (b_wide // pts // 100) * 100
            if unit >= 100:
                cost = unit * pts
                recommendations.append({
                    'bet_type': 'ワイド', 'type': 'ワイド', 'method': 'BOX',
                    'horse_numbers': top[:5], 'formation': [top[:5]],
                    'points': pts, 'unit_amount': unit, 'total_amount': cost,
                    'combination': "Top5 Box", 'reason': 'Flex Bal: Wide'
                })
        
        # 3. 3-Ren-Puku Box
        if b_trip >= 1000:
            pts = 10
            unit = (b_trip // pts // 100) * 100
            if unit >= 100:
                cost = unit * pts
                recommendations.append({
                    'bet_type': '3連複', 'type': '3連複', 'method': 'BOX',
                    'horse_numbers': top[:5], 'formation': [top[:5]],
                    'points': pts, 'unit_amount': unit, 'total_amount': cost,
                    'combination': "Top5 Box", 'reason': 'Flex Bal: Trip'
                })
        
        if not recommendations:
             return BettingAllocator.allocate_budget(df_preds, budget, strategy='balance')
             
        return recommendations

    @staticmethod
    def _allocate_wide_nagashi(df_preds: pd.DataFrame, budget: int) -> list:
        """
        低予算戦略1: ワイド軸1頭流し (400円〜500円)
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 5: return []
        
        axis = int(df_sorted.iloc[0]['horse_number'])
        opponents = df_sorted.iloc[1:6]['horse_number'].astype(int).tolist()
        
        # 予算に応じて相手数を調整 (500円未満なら4頭に絞る)
        if budget < 500:
            opponents = opponents[:4]
            
        points = len(opponents)
        cost = points * 100
        
        if cost > budget: return []
        
        return [{
            'type': 'ワイド', 'method': '流し',
            'horses': [axis] + opponents,
            'formation': [[axis], opponents],
            'amount': cost, 'count': points,
            'desc': '低予算ワイド流し'
        }]

    @staticmethod
    def _allocate_box4_umaren(df_preds: pd.DataFrame, budget: int) -> list:
        """
        低予算戦略2: 馬連4頭BOX (600円)
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 4: return []
        
        horses = df_sorted.iloc[:4]['horse_number'].astype(int).tolist()
        points = 6
        cost = 600
        
        if cost > budget: return []
        
        return [{
            'type': '馬連', 'method': 'BOX',
            'horses': horses, 'formation': [horses],
            'amount': cost, 'count': points,
            'desc': '馬連BOX4'
        }]

    @staticmethod
    def _allocate_box4_sanrenpuku(df_preds: pd.DataFrame, budget: int) -> list:
        """
        低予算戦略3: 3連複4頭BOX (400円)
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 4: return []
        
        horses = df_sorted.iloc[:4]['horse_number'].astype(int).tolist()
        points = 4
        cost = 400
        
        if cost > budget: return []
        
        return [{
            'type': '3連複', 'method': 'BOX',
            'horses': horses, 'formation': [horses],
            'amount': cost, 'count': points,
            'desc': '3連複BOX4'
        }]

    @staticmethod
    def _allocate_box5_sanrenpuku(df_preds: pd.DataFrame, budget: int) -> list:
        """
        三連複5頭BOX戦略 (5C3=10点, 1000円)
        AI予測上位5頭の組み合わせで高配当を狙う。
        4頭BOXより広い網を張り、的中率向上を目指す。
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 5:
            return []
        
        horses = df_sorted.iloc[:5]['horse_number'].astype(int).tolist()
        points = 10  # 5C3 = 10
        cost = points * 100  # 1000円
        
        if cost > budget:
            return []
        
        return [{
            'type': '3連複', 'method': 'BOX',
            'horses': horses, 'formation': [horses],
            'amount': cost, 'count': points,
            'desc': '3連複BOX5'
        }]

    @staticmethod
    def _allocate_umaren_nagashi(df_preds: pd.DataFrame, budget: int) -> list:
        """
        低予算戦略4: 馬連軸1頭流し (500円)
        - 1番人気から相手5頭へ流す
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 6: return []
        
        axis = int(df_sorted.iloc[0]['horse_number'])
        opponents = df_sorted.iloc[1:6]['horse_number'].astype(int).tolist() # Top 2-6 (5 horses)
        
        points = len(opponents)
        cost = points * 100
        
        if cost > budget:
             # Reduce to 4 points if budget < 500?
             if budget >= 400:
                 opponents = opponents[:4]
                 points = 4
                 cost = 400
             else:
                 return []
        
        return [{
            'type': '馬連', 'method': '流し',
            'horses': [axis] + opponents,
            'formation': [[axis], opponents],
            'amount': cost, 'count': points,
            'desc': '馬連軸流し'
        }]

    @staticmethod
    def _allocate_sanrenpuku_1axis(df_preds: pd.DataFrame, budget: int) -> list:
        """
        3連複1軸流し: 軸1頭 + 相手6頭 (15点 = 1500円)
        - Top1を軸に、Top2-7の6頭へ流す
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 7: return []
        
        axis = int(df_sorted.iloc[0]['horse_number'])
        opponents = df_sorted.iloc[1:7]['horse_number'].astype(int).tolist() # 6 horses
        
        # 6C2 = 15点
        points = len(opponents) * (len(opponents) - 1) // 2
        cost = points * 100
        
        if cost > budget:
             # 相手を5頭に絞る (5C2 = 10点 = 1000円)
             if budget >= 1000:
                 opponents = opponents[:5]
                 points = 10
                 cost = 1000
             else:
                 return []
        
        return [{
            'type': '3連複', 'method': '流し',
            'horses': [axis] + opponents,
            'formation': [[axis], opponents],
            'amount': cost, 'count': points,
            'desc': '3連複軸流し'
        }]

    @staticmethod
    def _allocate_sanrenpuku_2axis(df_preds: pd.DataFrame, budget: int) -> list:
        """
        3連複2軸流し: 軸2頭 + 相手5頭 (5点 = 500円)
        - Top1-2を軸に、Top3-7の5頭へ流す
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        if len(df_sorted) < 7: return []
        
        axis1 = int(df_sorted.iloc[0]['horse_number'])
        axis2 = int(df_sorted.iloc[1]['horse_number'])
        opponents = df_sorted.iloc[2:7]['horse_number'].astype(int).tolist() # 5 horses
        
        # 2軸流し = 相手頭数 (5点)
        points = len(opponents)
        cost = points * 100
        
        if cost > budget:
             return []
        
        return [{
            'type': '3連複', 'method': '流し', # 2軸流しだが流しとして処理
            'horses': [axis1, axis2] + opponents,
            'formation': [[axis1, axis2], opponents],
            'amount': cost, 'count': points,
            'desc': '3連複2軸流し'
        }]
    @staticmethod
    def _allocate_meta_optimized(df_preds: pd.DataFrame, budget: int) -> list:
        """
        メタ分析に基づく最適化戦略
        レーススコア(0-80)を算出し、スコアに応じて配分を動的調整
        """
        # 1. レース指標算出
        top_horse = df_preds.sort_values('probability', ascending=False).iloc[0]
        top1_prob = top_horse['probability']
        top1_odds = top_horse['odds']
        
        sorted_probs = df_preds['probability'].sort_values(ascending=False)
        prob_gap = sorted_probs.iloc[0] - sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0
        
        # 2. スコアリング (最大80点)
        score = 0
        
        # 低オッズボーナス (0-30点)
        if top1_odds < 5: score += 30
        elif top1_odds < 10: score += 20
        elif top1_odds < 15: score += 10
        
        # 高確率ボーナス (0-30点)
        if top1_prob > 0.9: score += 30
        elif top1_prob > 0.85: score += 20
        elif top1_prob > 0.7: score += 10
        
        # 過信ペナルティ (0 ~ -20点)
        if prob_gap > 0.7: score -= 20
        elif prob_gap > 0.5: score -= 10
        
        # 接戦ボーナス (0-20点)
        if 0.1 < prob_gap < 0.3: score += 20
        elif prob_gap < 0.1: score += 10
        
        # 3. 配分決定
        recs = []
        
        if score >= 60:
            # S: 最高条件 -> 3連複軸1頭流し (相手5-6頭)
            # 予算全額投入
            cand_horses = df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()
            axis = [cand_horses[0]]
            opponents = cand_horses[1:7] if len(cand_horses) >= 7 else cand_horses[1:]
            
            # nC2 * 100円 <= budget となるように相手を選ぶのが理想だが、
            # ここではシンプルに axis-opponent
            # 相手5頭=10点=1000円, 6頭=15点=1500円
            
            # 予算に合わせて相手数を調整
            n_opponents = 0
            for n in range(5, 12):
                cost = (n * (n-1) // 2) * 100
                if cost <= budget:
                    n_opponents = n
                else:
                    break
            
            if n_opponents >= 3:
                opponents = cand_horses[1:n_opponents+1]
                cost = (n_opponents * (n_opponents-1) // 2) * 100
                unit = budget // (cost // 100)
                
                recs.append({
                    'type': '3連複',
                    'method': 'NAGASHI',
                    'formation': [axis, opponents],
                    'amount': (cost // 100) * unit,
                    'unit_amount': unit,
                    'count': n_opponents * (n_opponents-1) // 2,
                    'desc': f'Sランク(Score:{score}): 3連複軸1頭流し',
                    'horses': axis + opponents
                })
                
        elif score >= 40:
            # A: 良好 -> 3連複BOX (予算80%) + ワイドBOX (予算20%)
            # 主力: 3連複5頭BOX (10点)
            total_budget = int(budget * 0.8 / 100) * 100
            if total_budget >= 1000:
                recs.append(BettingAllocator._create_box_rec('3連複', df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()[:5], total_budget))
                
            sub_budget = budget - (recs[0]['amount'] if recs else 0)
            if sub_budget >= 600:
                 recs.append(BettingAllocator._create_box_rec('ワイド', df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()[:4], sub_budget))
                 
        elif score >= 20: 
            # B: 標準 -> ワイドBOX (予算60%目安)
            target_budget = int(budget * 0.6 / 100) * 100
            if target_budget >= 600:
                recs.append(BettingAllocator._create_box_rec('ワイド', df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()[:5], target_budget))
                
        elif score >= 0:
            # C: 慎重 -> 馬連BOX (予算40%目安)
            target_budget = int(budget * 0.4 / 100) * 100
            if target_budget >= 600:
                recs.append(BettingAllocator._create_box_rec('馬連', df_preds.sort_values('probability', ascending=False)['horse_number'].tolist()[:4], target_budget))
        
        else:
            # D: スキップ
            return []
            
        return recs

    @staticmethod
    def _allocate_meta_contrarian(df_preds: pd.DataFrame, budget: int) -> list:
        """
        メタ分析に基づく逆張り戦略 (Contrarian)
        - スコアが低い（荒れる）時こそ、3連複BOX等で手広く高配当を狙う
        - スコアが高い（堅い）時は、3連単等で点数を絞って利益率を高める
        """
        # 1. レース指標算出
        top_horse = df_preds.sort_values('probability', ascending=False).iloc[0]
        top1_prob = top_horse['probability']
        top1_odds = top_horse['odds']
        
        sorted_probs = df_preds['probability'].sort_values(ascending=False)
        prob_gap = sorted_probs.iloc[0] - sorted_probs.iloc[1] if len(sorted_probs) > 1 else 0
        
        # 2. スコアリング (0-80点, 変動なし)
        score = 0
        if top1_odds < 5: score += 30
        elif top1_odds < 10: score += 20
        elif top1_odds < 15: score += 10
        
        if top1_prob > 0.9: score += 30
        elif top1_prob > 0.85: score += 20
        elif top1_prob > 0.7: score += 10
        
        if prob_gap > 0.7: score -= 20
        elif prob_gap > 0.5: score -= 10
        
        if 0.1 < prob_gap < 0.3: score += 20
        elif prob_gap < 0.1: score += 10
        
        # 3. 配分決定 (逆張りロジック)
        recs = []
        df_sorted = df_preds.sort_values('probability', ascending=False)
        horses = df_sorted['horse_number'].tolist()
        
        if score >= 60:
            # S: 鉄板 -> 3連単フォーメーション (1着固定 + ヒモ荒れ狙い)
            # 1着: Top1 (固定)
            # 2着: Top2-6 (5頭)
            # 3着: Top2-12 (11頭) - 3着に人気薄が飛び込むのを拾う
            # 点数: 1 * 5 * 10 - 重複 = 約40-50点?
            # Formation: [1], [2-6], [2-12]
            # 点数計算: 5 * 10 = 50点 (2着候補が3着にもいる場合) -> 実際は (5 * 11) - 5 (2着=3着のケース) = 50点
            
            if len(horses) >= 12:
                points = 50 
                # 予算5000円には収まる (5000円)
                
                # 予算調整
                cost = points * 100
                if cost > budget:
                    # 予算オーバーなら3着候補を削る
                    # 5000円未満の予算の場合など
                    n_3rd = 12
                    while (1 * 5 * (n_3rd - 2)) * 100 > budget and n_3rd > 6:
                        n_3rd -= 1
                    opponents_3rd = horses[1:n_3rd]
                else:
                    opponents_3rd = horses[1:12]
                
                points = 5 * (len(opponents_3rd) - 1)
                cost = points * 100
                unit = budget // (cost // 100)
                
                if unit >= 100:
                    recs.append({
                        'type': '3連単',
                        'method': 'FORMATION',
                        'formation': [[horses[0]], horses[1:6], opponents_3rd],
                        'amount': (cost // 100) * unit,
                        'unit_amount': unit,
                        'count': points,
                        'desc': f'Sランク(Score:{score}): 3連単1着不動・ヒモ荒れ狙い',
                        'horses': list(set([horses[0]] + horses[1:6] + opponents_3rd))
                    })
                    
        elif score >= 40:
            # A: 有力 -> 3連複 軸1頭流し (相手6頭 = 15点)
            # 軸: Top1, 相手: Top2-7
            if len(horses) >= 7:
                points = 15
                cost = points * 100
                unit = budget // (cost // 100)
                
                if unit >= 100:
                    recs.append({
                        'type': '3連複',
                        'method': 'NAGASHI',
                        'formation': [[horses[0]], horses[1:7]],
                        'amount': (cost // 100) * unit,
                        'unit_amount': unit,
                        'count': points,
                        'desc': f'Aランク(Score:{score}): 3連複軸1頭流し',
                        'horses': list(set([horses[0]] + horses[1:7]))
                    })
                    
        elif score >= 20: 
            # B: 混戦 -> 3連複 5頭BOX (10点)
            # 軸が信用できないためBOX
            if len(horses) >= 5:
                points = 10
                cost = points * 100
                unit = budget // (cost // 100)
                
                if unit >= 100:
                    recs.append({
                        'type': '3連複',
                        'method': 'BOX',
                        'formation': [horses[:5]],
                        'amount': (cost // 100) * unit,
                        'unit_amount': unit,
                        'count': points,
                        'desc': f'Bランク(Score:{score}): 3連複5頭BOX',
                        'horses': horses[:5]
                    })
                
        else:
            # C: 大荒れ (Score < 20) -> 3連複 6-7頭BOX (20点 or 35点)
            # 紛れ当たりを狙って広く構える
            # 予算5000円なら7頭BOX(3500円)が可能
            
            box_size = 7
            points = 35 # 7C3
            
            # 予算不足なら6頭(20点)に縮小
            if points * 100 > budget:
                box_size = 6
                points = 20
                
            if len(horses) >= box_size:
                cost = points * 100
                unit = budget // (cost // 100)
                
                if unit >= 100:
                    recs.append({
                        'type': '3連複',
                        'method': 'BOX',
                        'formation': [horses[:box_size]],
                        'amount': (cost // 100) * unit,
                        'unit_amount': unit,
                        'count': points,
                        'desc': f'Cランク(Score:{score}): 3連複{box_size}頭BOX (波乱狙い)',
                        'horses': horses[:box_size]
                    })
        
        return recs
    
    @staticmethod
    def _allocate_ranking_anchor(df_preds: pd.DataFrame, budget: int) -> list:
        """
        ランキング学習モデル(LTR)専用戦略: Ranking Anchor
        - 指標: probability (LTRスコア)
        - 軸馬(1位)の複勝率の高さを活用
        - スコア差(Gap)に応じて配分を調整
        """
        df_sorted = df_preds.sort_values('probability', ascending=False)
        horses = df_sorted['horse_number'].tolist()
        probs = df_sorted['probability'].tolist()
        
        if len(horses) < 6:
            return BettingAllocator._allocate_formation(df_preds, budget) # Fallback to formation
            
        axis = horses[0]
        gap = probs[0] - probs[1] if len(probs) > 1 else 0
        
        recs = []
        remaining_budget = budget
        
        # 1. 単勝保険 (Gapが大きい場合のみ)
        if gap > 0.05 and remaining_budget >= 500:
            win_amt = min(1000, int(remaining_budget * 0.2 / 100) * 100)
            if win_amt >= 100:
                recs.append({
                    'type': '単勝', 'method': 'SINGLE', 'horses': [axis],
                    'formation': [[axis]], 'amount': win_amt, 'count': 1,
                    'desc': f'LTR単勝(Gap:{gap:.2f})'
                })
                remaining_budget -= win_amt
                
        # 2. ワイド軸1頭流し (的中率維持)
        # 相手は2-5位の4頭
        wide_opps = horses[1:5]
        wide_pts = len(wide_opps)
        wide_budget = int(remaining_budget * 0.4 / 100) * 100
        if wide_budget >= wide_pts * 100:
            unit = wide_budget // wide_pts
            recs.append({
                'type': 'ワイド', 'method': '流し', 'horses': [axis] + wide_opps,
                'formation': [[axis], wide_opps], 'amount': (wide_budget // 100) * 100,
                'unit_amount': unit, 'count': wide_pts, 'desc': 'LTRワイド軸流し'
            })
            remaining_budget -= (wide_budget // 100) * 100
            
        # 3. 三連複軸1頭流し (高配当狙い)
        # 相手は2-7位の6頭 (15点)
        trip_opps = horses[1:7]
        trip_pts = 15 # 6C2
        if remaining_budget >= trip_pts * 100:
            unit = remaining_budget // trip_pts
            recs.append({
                'type': '3連複', 'method': '流し', 'horses': [axis] + trip_opps,
                'formation': [[axis], trip_opps], 'amount': (remaining_budget // 100) * 100,
                'unit_amount': unit, 'count': trip_pts, 'desc': 'LTR三連複軸流し'
            })
            remaining_budget = 0
            
        return recs
