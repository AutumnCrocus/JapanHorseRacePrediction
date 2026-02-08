import pandas as pd
import numpy as np

class BettingStrategy:
    """馬券戦略・予算配分クラス"""
    
    @staticmethod
    def generate_reason_from_shap(reasoning: dict) -> str:
        """
        SHAP値の情報から日本語の判断根拠を生成する
        """
        if not reasoning:
            return ""
            
        positive = reasoning.get('positive', [])
        if not positive:
            return ""
            
        # 特徴量ラベルのマッピング (代表的なもの)
        labels = {
            'avg_last_3f': '末脚',
            'jockey_win_rate': '名手',
            'win_rate': '勝率',
            'place_rate': '複勝率',
            'avg_rank': '安定感',
            'popularity': '実力',
            '単勝': '期待値',
            '期待値': '期待値',
            '斤量': '斤量',
            '年齢': '若さ',
            '体重変化': '馬体',
            'course_len': '適性',
            'race_count': '経験'
        }
        
        # 上位3つのプラス要因を抽出
        factors = []
        for p in positive[:3]:
            feat = p.get('feature', '')
            label = labels.get(feat, feat)
            if label:
                factors.append(label)
        
        if factors:
            # 重複削除
            factors = list(dict.fromkeys(factors))
            return "、".join(factors) + "がプラス"
            
        return ""

    @staticmethod
    def generate_reason(bet_type: str, horses: list, prob: float, ev: float, odds: float, features_list: list = None, reasoning: dict = None) -> str:
        """
        購入理由を日本語で生成 (特徴量に基づく詳細版)
        """
        feature_msg = []
        
        # 1. SHAP値（reasoning）がある場合は最優先で使用
        shap_msg = BettingStrategy.generate_reason_from_shap(reasoning)
        if shap_msg:
            feature_msg.append(shap_msg)
        
        # 2. 従来の特徴量ベースの簡易判定 (SHAPがない場合のバックアップ)
        if not shap_msg and features_list and len(features_list) > 0:
            feat = features_list[0]
            
            # 末脚 (avg_last_3f)
            last_3f = feat.get('avg_last_3f', 37.0)
            if last_3f < 34.5:
                feature_msg.append("鋭い末脚")
            elif last_3f < 35.0:
                feature_msg.append("安定した末脚")
                
            # 騎手 (jockey_win_rate)
            j_rate = feat.get('jockey_win_rate', 0.0)
            if j_rate > 0.15:
                feature_msg.append("名手とのコンビ")
            elif j_rate > 0.10:
                feature_msg.append("実績ある騎手")
                
            # 人気と実力のギャップ
            pop = feat.get('popularity', 0)
            if prob > 0.3 and pop > 3:
                feature_msg.append("実力過小評価")
            elif pop > 5:
                feature_msg.append("穴妙味あり")

        # 結合して「〜で、〜」の形にする
        extra_text = ""
        if feature_msg:
            extra_text = "、".join(feature_msg) + "。 "

        if bet_type == 'tan':
            if ev > 2.0:
                return f"{extra_text}期待値{ev:.2f}倍の本命"
            elif prob > 0.5:
                return f"{extra_text}勝率{prob*100:.0f}%と盤石"
            else:
                return f"{extra_text}期待値{ev:.2f}倍で狙い目" if ev > 0.1 else f"{extra_text}穴狙い"
                
        elif bet_type == 'fuku':
            if prob > 0.8:
                return f"{extra_text}複勝率{prob*100:.0f}%と鉄板"
            elif ev > 1.5:
                return f"{extra_text}期待値{ev:.2f}倍の好走期待"
            else:
                return f"{extra_text}複勝圏内有力"
                
        elif bet_type == 'umaren':
            if ev > 0.1:
                return f"期待値{ev:.2f}倍。{extra_text}上位拮抗" if extra_text else f"上位2頭の組み合わせで期待値{ev:.2f}倍"
            else:
                return f"{extra_text}上位拮抗" if extra_text else "上位2頭の組み合わせ推奨"
                
        elif bet_type == 'wide':
            if ev > 0.1:
                return f"{extra_text}手堅く期待値{ev:.2f}倍" if extra_text else f"的中率重視で期待値{ev:.2f}倍"
            else:
                return f"{extra_text}手堅く的中狙い" if extra_text else "的中率重視で推奨"
                
        elif bet_type == 'umatan':
            return f"{extra_text}着順通りの決着なら高配当"
            
        elif bet_type == 'sanrenpuku':
            if ev > 0.1:
                return f"期待値{ev:.2f}倍。{extra_text}3頭の好走に期待"
            else:
                return f"{extra_text}3頭の好走に期待" if extra_text else "3頭の好走に期待"
                
        elif bet_type == 'sanrentan':
            if odds > 0:
                return f"一撃高配当狙い（{odds:.1f}倍）"
            else:
                return "一撃高配当狙い"
        
        return f"期待値{ev:.2f}倍" if ev > 0.1 else "推奨"

    @staticmethod
    def generate_box_reason(bet_type: str, horse_indices: list, df_preds: pd.DataFrame, odds_data: dict = None) -> str:
        """
        BOX買い用の理由生成 (実際の組み合わせ期待値を算出)
        """
        # 対象馬のデータを抽出
        target_indices = [int(h) for h in horse_indices]
        target_rows = df_preds[df_preds['horse_number'].isin(target_indices)]
        
        if target_rows.empty:
            return "ボックス推奨"

        # オッズ平均の計算（期待値ではなく馬券オッズの平均）
        total_odds_sum = 0.0
        valid_odds_count = 0
        
        if odds_data:
            import itertools
            
            # 券種に応じたキーとオッズ辞書
            odds_dict = {}
            combos = []
            type_key = ''
            
            if bet_type == '馬連':
                type_key = 'umaren'
                odds_dict = odds_data.get('umaren', {})
                combos = list(itertools.combinations(target_indices, 2))
            elif bet_type == 'ワイド':
                type_key = 'wide'
                odds_dict = odds_data.get('wide', {})
                combos = list(itertools.combinations(target_indices, 2))
            elif bet_type == '3連複':
                type_key = 'sanrenpuku'
                odds_dict = odds_data.get('sanrenpuku', {})
                combos = list(itertools.combinations(target_indices, 3))
            elif bet_type == '3連単':
                type_key = 'sanrentan'
                odds_dict = odds_data.get('sanrentan', {})
                combos = list(itertools.permutations(target_indices, 3))
            
            # オッズを集計
            for c in combos:
                # Key作成 (ソート済みタプルまたはそのまま)
                key = c
                if bet_type in ['馬連', 'ワイド', '3連複']:
                    key = tuple(sorted(c))
                
                # オッズ取得
                odds = 0
                if key in odds_dict:
                    val = odds_dict[key]
                    # ワイドの場合は[min, max]なので平均をとるかminをとる
                    if isinstance(val, list):
                        odds = val[0]
                    else:
                        odds = val
                
                if odds > 0:
                    total_odds_sum += odds
                    valid_odds_count += 1

        # 平均オッズを計算
        avg_odds = 0
        if valid_odds_count > 0:
            avg_odds = total_odds_sum / valid_odds_count
        else:
            # オッズデータがない場合は単勝EVの平均で代用
            avg_odds = target_rows['expected_value'].mean()
        
        # メッセージ生成
        reason_parts = []
        
        # 1. オッズ
        if avg_odds > 0:
            reason_parts.append(f"平均オッズ{avg_odds:.1f}倍")
            
        # 2. 定性評価
        has_ana = any((target_rows['odds'] > 20) & (target_rows['probability'] > 0.05))
        is_solid = all(target_rows['probability'] > 0.15)
        
        if has_ana:
            reason_parts.append("高配当狙い")
        elif is_solid:
            reason_parts.append("堅実構成")
        else:
            reason_parts.append("バランス型")
            
        return "。".join(reason_parts)
    
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
        horse_features = {}
        for _, row in df.iterrows():
            umaban = int(row.get('horse_number', row.get('馬番', 0)))
            if umaban > 0:
                horse_probs[umaban] = row.get('probability', 0.0)
                horse_names[umaban] = row.get('horse_name', row.get('馬名', '不明'))
                horse_features[umaban] = row.to_dict()
        
        # 単勝の推奨計算
        if 'tan' in odds_data and odds_data['tan']:
            tan_odds = odds_data['tan']
            for umaban, prob in horse_probs.items():
                if umaban in tan_odds:
                    odds = tan_odds[umaban]
                    ev = prob * odds
                    
                    if ev > 0.5:
                        features = [horse_features[umaban]] if umaban in horse_features else []
                        reason = BettingStrategy.generate_reason('tan', [horse_names[umaban]], prob, ev, odds, features)
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
                        features = [horse_features[umaban]] if umaban in horse_features else []
                        reason = BettingStrategy.generate_reason('fuku', [horse_names[umaban]], fuku_prob, ev, min_odds, features)
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
                            # スコアが高い方の馬の特徴を優先
                            feat1 = horse_features.get(uma1, {})
                            feat2 = horse_features.get(uma2, {})
                            main_feat = feat1 if prob1 > prob2 else feat2
                            reason = BettingStrategy.generate_reason('umaren', horses_list, combo_prob, ev, odds, [main_feat])
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
                            feat1 = horse_features.get(uma1, {})
                            feat2 = horse_features.get(uma2, {})
                            main_feat = feat1 if prob1 > prob2 else feat2
                            reason = BettingStrategy.generate_reason('wide', horses_list, combo_prob, ev, min_odds, [main_feat])
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
                                # 最も確率が高い馬の特徴を採用
                                best_uma = max([(uma1, prob1), (uma2, prob2), (uma3, prob3)], key=lambda x: x[1])[0]
                                reason = BettingStrategy.generate_reason('sanrenpuku', horses_list, combo_prob, ev, odds, [horse_features.get(best_uma, {})])
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

    @staticmethod
    def optimize_allocation_kelly(recommendations: pd.DataFrame, budget: int, 
                                  use_half_kelly: bool = True,
                                  min_edge: float = 0.05,
                                  max_fraction: float = 0.25) -> list:
        """
        Kelly基準による予算配分
        
        Args:
            recommendations: 推奨DataFrame (prob, odds, ev列必須)
            budget: 総予算
            use_half_kelly: Half-Kelly（リスク軽減版）を使用するか
            min_edge: 最小エッジ閾値（これ以下は賭けない）
            max_fraction: 1ベット最大比率
            
        Returns:
            配分済み推奨リスト
        """
        if recommendations.empty or budget <= 0:
            return []
        
        # 期待値(ev)で降順ソート
        df = recommendations.sort_values('ev', ascending=False)
        
        results = []
        total_allocated = 0
        
        for _, row in df.iterrows():
            prob = row.get('prob', 0)
            odds = row.get('odds', 0)
            
            if prob <= 0 or odds <= 1:
                continue
            
            # 期待値チェック
            ev = prob * odds
            if ev < 1 + min_edge:
                continue
            
            # Kelly計算: f* = (p*b - q) / b
            b = odds - 1
            q = 1 - prob
            fraction = (prob * b - q) / b
            
            if fraction <= 0:
                continue
            
            # Half-Kelly適用
            if use_half_kelly:
                fraction *= 0.5
            
            # 最大比率でキャップ
            fraction = min(fraction, max_fraction)
            
            # 賭け金計算（100円単位）
            amount = int(budget * fraction / 100) * 100
            
            if amount < 100:
                continue
            
            results.append({
                'type': row['type'],
                'combination': row.get('combination', str(row.get('umaban', ''))),
                'umaban': row.get('umaban', 0),
                'name': row.get('name', ''),
                'odds': odds,
                'prob': prob,
                'ev': ev,
                'kelly_fraction': fraction,
                'amount': amount,
                'return': int(amount * odds),
                'reason': f"Kelly推奨: 期待値{ev:.2f}倍, 配分{fraction*100:.1f}%"
            })
            
            total_allocated += amount
        
        # 予算オーバーの場合は比率で調整
        if total_allocated > budget and results:
            scale = budget / total_allocated
            for r in results:
                r['amount'] = max(100, int(r['amount'] * scale / 100) * 100)
                r['return'] = int(r['amount'] * r['odds'])
        
        return results
