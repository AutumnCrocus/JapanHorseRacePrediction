import pandas as pd
import numpy as np

class BettingStrategy:
    """馬券戦略・予算配分クラス"""
    
    @staticmethod
    def generate_reason(bet_type: str, horses: list, prob: float, ev: float, odds: float) -> str:
        """
        購入理由を日本語で生成
        
        Args:
            bet_type: 券種 ('tan', 'fuku', 'umaren', etc.)
            horses: 馬名のリスト
            prob: 的中確率
            ev: 期待値
            odds: オッズ
            
        Returns:
            購入理由の文字列
        """
        if bet_type == 'tan':
            if ev > 1.5:
                return f"高い期待値{ev:.2f}倍で本命候補"
            elif prob > 0.6:
                return f"的中確率{prob*100:.1f}%で堅実"
            else:
                return f"期待値{ev:.2f}倍で妙味あり"
                
        elif bet_type == 'fuku':
            if prob > 0.8:
                return f"複勝率{prob*100:.1f}%と堅実"
            elif ev > 1.5:
                return f"期待値{ev:.2f}倍で妙味あり"
            else:
                return f"複勝圏内の可能性が高い"
                
        elif bet_type == 'umaren':
            if len(horses) == 2:
                return f"上位2頭の組み合わせで期待値{ev:.2f}倍"
            else:
                return f"期待値{ev:.2f}倍で推奨"
                
        elif bet_type == 'wide':
            if prob > 0.5:
                return f"両馬とも上位入線の可能性が高く期待値{ev:.2f}倍"
            else:
                return f"穴狙いで高配当が期待できる"
                
        elif bet_type == 'umatan':
            return f"着順予想通りなら{odds:.1f}倍の配当"
            
        elif bet_type == 'sanrenpuku':
            if prob > 0.3:
                return f"TOP3候補の組み合わせで期待値{ev:.2f}倍"
            else:
                return f"高配当狙いで{odds:.1f}倍"
                
        elif bet_type == 'sanrentan':
            return f"着順的中で{odds:.1f}倍の高配当が期待できる"
        
        return f"期待値{ev:.2f}倍"
    
    @staticmethod
    def calculate_expected_value(predictions: pd.DataFrame, odds_data: dict) -> pd.DataFrame:
        """
        予測結果とオッズから期待値を計算して推奨度を付与
        """
        df = predictions.copy()
        recommendations = []
        
        # 馬番と確率のマッピングを作成
        horse_probs = {}
        horse_names = {}
        for _, row in df.iterrows():
            umaban = int(row.get('horse_number', row.get('馬番', 0)))
            if umaban > 0:
                horse_probs[umaban] = row.get('probability', 0.0)
                horse_names[umaban] = row.get('horse_name', row.get('馬名', '不明'))
        
        # 単勝の推奨計算
        if 'tan' in odds_data and odds_data['tan']:
            tan_odds = odds_data['tan']
            for umaban, prob in horse_probs.items():
                if umaban in tan_odds:
                    odds = tan_odds[umaban]
                    ev = prob * odds
                    
                    if ev > 0.5:
                        reason = BettingStrategy.generate_reason('tan', [horse_names[umaban]], prob, ev, odds)
                        recommendations.append({
                            'type': '単勝',
                            'type_code': 'tan',
                            'combination': f"{umaban}",
                            'umaban': umaban,
                            'name': horse_names[umaban],
                            'odds': odds,
                            'prob': prob,
                            'ev': ev,
                            'score': ev * prob,
                            'reason': reason
                        })
                        
        # 複勝の推奨計算
        if 'fuku' in odds_data and odds_data['fuku']:
            fuku_odds = odds_data['fuku']
            for umaban, prob in horse_probs.items():
                # 複勝確率は単勝確率の3倍程度（簡易推定）
                fuku_prob = min(prob * 3.0, 0.95)
                
                if umaban in fuku_odds:
                    min_odds, max_odds = fuku_odds[umaban]
                    ev = fuku_prob * min_odds
                    
                    if ev > 0.8:
                        reason = BettingStrategy.generate_reason('fuku', [horse_names[umaban]], fuku_prob, ev, min_odds)
                        recommendations.append({
                            'type': '複勝',
                            'type_code': 'fuku',
                            'combination': f"{umaban}",
                            'umaban': umaban,
                            'name': horse_names[umaban],
                            'odds': min_odds,
                            'prob': fuku_prob,
                            'ev': ev,
                            'score': ev * fuku_prob,
                            'reason': reason
                        })
        
        # 馬連の推奨計算（上位馬の組み合わせ）
        if 'umaren' in odds_data and odds_data['umaren']:
            top_horses = sorted(horse_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (uma1, prob1) in enumerate(top_horses):
                for uma2, prob2 in top_horses[i+1:]:
                    key = (uma1, uma2) if uma1 < uma2 else (uma2, uma1)
                    if key in odds_data['umaren']:
                        odds = odds_data['umaren'][key]
                        # 組み合わせの確率（簡易計算）
                        combo_prob = prob1 * prob2 * 0.5  # 両馬が上位にくる確率を推定
                        ev = combo_prob * odds
                        
                        if ev > 1.5:  # 期待値の閖値
                            horses_list = [horse_names[uma1], horse_names[uma2]]
                            reason = BettingStrategy.generate_reason('umaren', horses_list, combo_prob, ev, odds)
                            recommendations.append({
                                'type': '馬連',
                                'type_code': 'umaren',
                                'combination': f"{uma1}-{uma2}",
                                'umaban': f"{uma1}-{uma2}",
                                'name': f"{horse_names[uma1]}/{horse_names[uma2]}",
                                'odds': odds,
                                'prob': combo_prob,
                                'ev': ev,
                                'score': ev * combo_prob,
                                'reason': reason
                            })
        
        # ワイドの推奨計算
        if 'wide' in odds_data and odds_data['wide']:
            top_horses = sorted(horse_probs.items(), key=lambda x: x[1], reverse=True)[:6]
            
            for i, (uma1, prob1) in enumerate(top_horses):
                for uma2, prob2 in top_horses[i+1:]:
                    key = (uma1, uma2) if uma1 < uma2 else (uma2, uma1)
                    if key in odds_data['wide']:
                        min_odds, max_odds = odds_data['wide'][key]
                        # 3着以内に両馬が入る確率（簡易計算）
                        combo_prob = prob1 * prob2 * 0.8  # 両馬が3着以内に入る確率を推定
                        ev = combo_prob * min_odds
                        
                        if ev > 1.2:
                            horses_list = [horse_names[uma1], horse_names[uma2]]
                            reason = BettingStrategy.generate_reason('wide', horses_list, combo_prob, ev, min_odds)
                            recommendations.append({
                                'type': 'ワイド',
                                'type_code': 'wide',
                                'combination': f"{uma1}-{uma2}",
                                'umaban': f"{uma1}-{uma2}",
                                'name': f"{horse_names[uma1]}/{horse_names[uma2]}",
                                'odds': min_odds,
                                'prob': combo_prob,
                                'ev': ev,
                                'score': ev * combo_prob,
                                'reason': reason
                            })
        
        # 三連複の推奨計算（上位馬のみ）
        if 'sanrenpuku' in odds_data and odds_data['sanrenpuku']:
            top_horses = sorted(horse_probs.items(), key=lambda x: x[1], reverse=True)[:4]
            
            for i, (uma1, prob1) in enumerate(top_horses):
                for j, (uma2, prob2) in enumerate(top_horses[i+1:], i+1):
                    for uma3, prob3 in top_horses[j+1:]:
                        key = tuple(sorted([uma1, uma2, uma3]))
                        if key in odds_data['sanrenpuku']:
                            odds = odds_data['sanrenpuku'][key]
                            # 3馬が3着以内に入る確率（簡易計算）
                            combo_prob = prob1 * prob2 * prob3 * 0.3  # と3馬が3着以内に入る確率を推定
                            ev = combo_prob * odds
                            
                            if ev > 2.0:
                                horses_list = [horse_names[uma1], horse_names[uma2], horse_names[uma3]]
                                reason = BettingStrategy.generate_reason('sanrenpuku', horses_list, combo_prob, ev, odds)
                                recommendations.append({
                                    'type': '三連複',
                                    'type_code': 'sanrenpuku',
                                    'combination': f"{uma1}-{uma2}-{uma3}",
                                    'umaban': f"{uma1}-{uma2}-{uma3}",
                                    'name': f"{horse_names[uma1]}/{horse_names[uma2]}/{horse_names[uma3]}",
                                    'odds': odds,
                                    'prob': combo_prob,
                                    'ev': ev,
                                    'score': ev * combo_prob,
                                    'reason': reason
                                })

        return pd.DataFrame(recommendations)

    @staticmethod
    def optimize_allocation(recommendations: pd.DataFrame, budget: int) -> list:
        """
        予算配分の最適化 (簡易的な比例配分ロジック)
        """
        if recommendations.empty or budget <= 0:
            return []
            
        # スコアで降順ソート
        df = recommendations.sort_values('score', ascending=False).head(10)  # 上位10個に拡張
        
        total_score = df['score'].sum()
        results = []
        
        remaining_budget = budget
        
        for _, row in df.iterrows():
            # 予算配分 (100円単位で切り捨て)
            allocation_ratio = row['score'] / total_score
            amount = int((budget * allocation_ratio) / 100) * 100
            
            if amount < 100:
                amount = 0
            
            if amount > 0:
                results.append({
                    'type': row['type'],
                    'combination': row.get('combination', str(row['umaban'])),
                    'umaban': row['umaban'],
                    'name': row['name'],
                    'odds': row['odds'],
                    'prob': row['prob'],
                    'ev': row['ev'],
                    'amount': amount,
                    'return': int(amount * row['odds']),
                    'reason': row.get('reason', '')
                })
                remaining_budget -= amount
                
        # 余り予算があれば一番期待値が高いものに追加
        if remaining_budget >= 100 and results:
            results[0]['amount'] += remaining_budget
            results[0]['return'] = int(results[0]['amount'] * results[0]['odds'])
            
        return results
