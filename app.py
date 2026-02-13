"""
競馬予想AI - Webアプリケーション
Flask APIサーバーとUI (Last Updated: Phase 3 Optimization v9 - Integrated Extreme Speed IPAT Automation)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime

# モジュールのインポート（絶対インポート）
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.constants import MODEL_DIR
from modules.training import HorseRaceModel, create_sample_model, EnsembleModel
from modules.scraping import Odds
from modules.strategy import BettingStrategy

app = Flask(__name__, static_folder='static', template_folder='templates')

# グローバル変数でモデルを保持 (複数モデル対応)
# { 'lgbm': HorseRaceModel, 'ltr': RankingWrapper }
MODELS = {}
PROCESSORS = {}
ENGINEERS = {}
bias_map = None
jockey_stats = None

# レース予測結果のキャッシュ (メモリ保持)
# { race_id: { 'df': DataFrame, 'results': list, 'timestamp': datetime } }
PREDICTION_CACHE = {}

def get_model(model_type='lgbm'):
    """指定された種類のモデルを取得（未ロードならロードする）"""
    if model_type not in MODELS:
        load_model(model_type)
    return MODELS.get(model_type)


def load_model(model_type='lgbm'):
    """モデルを読み込み（LGBM または LTR）"""
    global bias_map, jockey_stats
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 共通のメタデータ読み込み
    bias_map_path = os.path.join(MODEL_DIR, 'bias_map.pkl')
    jockey_stats_path = os.path.join(MODEL_DIR, 'jockey_stats.pkl')
    
    if bias_map is None and os.path.exists(bias_map_path):
        with open(bias_map_path, 'rb') as f:
            bias_map = pickle.load(f)
    if jockey_stats is None and os.path.exists(jockey_stats_path):
        try:
            with open(jockey_stats_path, 'rb') as f:
                jockey_stats = pickle.load(f)
        except:
            jockey_stats = None

    if model_type == 'lgbm':
        # 従来の LGBM モデル
        latest_model_dir = os.path.join(MODEL_DIR, 'historical_2010_2026')
        latest_model_path = os.path.join(latest_model_dir, 'model.pkl')
        
        if os.path.exists(latest_model_path):
            print(f"LGBMモデルを読み込み中: {latest_model_path}")
            m = HorseRaceModel()
            m.load(latest_model_path)
            MODELS['lgbm'] = m
            
            # ProcessorとEngineer
            proc_path = os.path.join(latest_model_dir, 'processor.pkl')
            eng_path = os.path.join(latest_model_dir, 'engineer.pkl')
            if os.path.exists(proc_path):
                with open(proc_path, 'rb') as f:
                    PROCESSORS['lgbm'] = pickle.load(f)
            if os.path.exists(eng_path):
                with open(eng_path, 'rb') as f:
                    ENGINEERS['lgbm'] = pickle.load(f)
            print("LGBMモデルのロード完了。")

    elif model_type == 'ltr':
        # 新しい LTR モデル
        ltr_model_dir = os.path.join(MODEL_DIR, 'standalone_ranking')
        ltr_model_path = os.path.join(ltr_model_dir, 'ranking_model.pkl')
        
        if os.path.exists(ltr_model_path):
            print(f"LTRモデルを読み込み中: {ltr_model_path}")
            with open(ltr_model_path, 'rb') as f:
                data = pickle.load(f)
            
            # 簡易ラッパー
            class RankingWrapper:
                def __init__(self, data):
                    self.model = data['model']
                    self.feature_names = data['feature_names']
                    self.model_type = 'ltr'
                def predict(self, X):
                    return self.model.predict(X[self.feature_names])
                def get_feature_importance(self, top_n=15):
                    importances = self.model.feature_importance(importance_type='gain')
                    return pd.DataFrame({'feature': self.feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(top_n)
                def debug_info(self):
                    return {'model_type': 'ltr', 'feature_names': self.feature_names}

            MODELS['ltr'] = RankingWrapper(data)
            
            # ProcessorとEngineer (LGBMと共有または独自。一旦LGBMのものを流用)
            # スタンドアロン作成時に上位モデルと構成を合わせているため、LGBM版が使える
            latest_model_dir = os.path.join(MODEL_DIR, 'historical_2010_2026')
            proc_path = os.path.join(latest_model_dir, 'processor.pkl')
            eng_path = os.path.join(latest_model_dir, 'engineer.pkl')
            if os.path.exists(proc_path):
                with open(proc_path, 'rb') as f:
                    PROCESSORS['ltr'] = pickle.load(f)
            if os.path.exists(eng_path):
                with open(eng_path, 'rb') as f:
                    ENGINEERS['ltr'] = pickle.load(f)
            print("LTRモデルのロード完了。")


@app.route('/')
def index():
    """メインページ"""
    # デフォルトモデル(LGBM)の情報を取得
    model = get_model('lgbm')
    model_data = {
        'success': False,
        'algorithm': 'LightGBM (Historical)',
        'last_updated': '-',
        'feature_count': 0,
        'features': [],
        'available': False,
        'metrics': {'auc': 0.802, 'recovery_rate': 114.1}
    }
    
    if model:
        try:
            latest_model_dir = os.path.join(MODEL_DIR, 'historical_2010_2026')
            model_path = os.path.join(latest_model_dir, 'model.pkl')
            
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                dt = datetime.fromtimestamp(mtime)
                model_data['last_updated'] = dt.strftime('%Y/%m/%d %H:%M')

            model_data['feature_count'] = len(model.feature_names) if model.feature_names else 0
            model_data['target'] = '複勝（3着以内）'
            model_data['source'] = 'netkeiba.com'
            model_data['success'] = True

            importance = model.get_feature_importance(15)
            model_data['available'] = True
            model_data['features'] = [{'feature': str(row['feature']), 'importance': float(row['importance'])} for _, row in importance.iterrows()]
                
        except Exception as e:
            print(f"Server-side data injection error: {e}")

    return render_template('index.html', initial_model_data=json.dumps(model_data, ensure_ascii=False))


from modules.data_loader import fetch_and_process_race_data
import re

@app.route('/api/predict', methods=['POST'])
def predict():
    """予測API"""
    try:
        data = request.json
        model_type = data.get('model_type', 'lgbm')
        model = get_model(model_type)
        
        if model is None:
            return jsonify({'error': f'モデル({model_type})の読み込みに失敗しました'}), 500
        
        horses = data.get('horses', [])
        if not horses:
            return jsonify({'error': '馬データが必要です'}), 400
        
        df = pd.DataFrame(horses)
        budget = int(data.get('budget', 0))
        race_id = data.get('race_id', 'custom_race')
        strategy = data.get('strategy', 'balance')
        
        return run_prediction_logic(df, "カスタムレース", "入力データによる予測", race_id=race_id, budget=budget, strategy=strategy, model_type=model_type)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_by_url', methods=['POST'])
def predict_by_url():
    """URLから予測"""
    try:
        data = request.json
        url = data.get('url', '')
        model_type = data.get('model_type', 'lgbm')
        
        if not url:
            return jsonify({'error': 'URLが必要です'}), 400
            
        match = re.search(r'race_id=(\d+)', url)
        if not match:
            return jsonify({'error': '有効なNetkeibaのレースURLではありません'}), 400
            
        race_id = match.group(1)
        
        try:
            # キャッシュのチェック (model_typeも考慮)
            cache_key = f"{race_id}_{model_type}"
            if cache_key in PREDICTION_CACHE:
                print(f"Using cached results for key: {cache_key}")
                # キャッシュがあればそのまま返すロジックを呼ぶか、run_prediction_logicに任せる
                # run_prediction_logic内でキャッシュチェックを行うように修正
            
            # データ取得 (Processor/Engineer は選択されたモデルのものを使用)
            proc = PROCESSORS.get(model_type)
            eng = ENGINEERS.get(model_type)
            df = fetch_and_process_race_data(race_id, proc, eng, bias_map, jockey_stats)
        except Exception as e:
            return jsonify({'error': f'データ取得に失敗しました: {str(e)}'}), 500
            
        if df.empty:
            return jsonify({'error': 'データが見つかりませんでした'}), 400
            
        # 予測実行
        budget_raw = data.get('budget', 0)
        budget = int(budget_raw) if budget_raw not in (None, '') else 0
        strategy = data.get('strategy', 'balance')
        
        return run_prediction_logic(df, f"Netkeiba Race {race_id}", "URLからの取得データ", race_id=race_id, budget=budget, strategy=strategy, model_type=model_type)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

from modules.betting_allocator import BettingAllocator

def run_prediction_logic(df, race_name_default, race_info_default, race_id=None, budget=0, strategy='balance', model_type='lgbm'):
    """共通予測ロジック"""
    
    # キャッシュのチェック (model_typeも考慮)
    cache_key = f"{race_id}_{model_type}" if race_id else None
    results = None
    if cache_key and cache_key in PREDICTION_CACHE and 'results' in PREDICTION_CACHE[cache_key]:
        print(f"Using cached inference results for key: {cache_key}")
        cached_data = PREDICTION_CACHE[cache_key]
        results = cached_data['results']
        race_name = cached_data.get('race_name', race_name_default)
        race_data01 = cached_data.get('race_data01', '')
        race_data02 = cached_data.get('race_data02', '')
    
    if results is None:
        # キャッシュがない場合は通常通り推論を実行
        model = get_model(model_type)
        if model is None:
            return jsonify({'error': f'モデル({model_type})の読み込みに失敗しました'}), 500
        
        feature_names = model.feature_names
        
        # 欠損している特徴量はデフォルト値で埋める (Simplified for brevity)
        for col in feature_names:
            if col not in df.columns:
                if col in ['枠番', '馬番', '人気']: df[col] = df.index + 1
                elif col in ['斤量']: df[col] = 56.0
                elif col in ['単勝']: df[col] = 10.0
                elif col in ['年齢']: df[col] = 4
                elif col in ['体重']: df[col] = 480
                elif col in ['体重変化']: df[col] = 0
                elif col in ['course_len']: df[col] = 2000
                elif col in ['avg_rank']: df[col] = 5.0
                elif col in ['win_rate']: df[col] = 0.1
                elif col in ['place_rate']: df[col] = 0.3
                elif col in ['race_count']: df[col] = 10
                elif col in ['jockey_avg_rank']: df[col] = 5.0
                elif col in ['jockey_win_rate']: df[col] = 0.1
                elif col in ['性', 'race_type']: df[col] = 0
                elif col in ['avg_last_3f']: df[col] = 37.0
                elif col in ['avg_running_style']: df[col] = 0.5
                elif col in ['venue_id', 'kai', 'day', 'race_num']: df[col] = 0
                else: df[col] = 0
        
        X = df[feature_names].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X = X.fillna(0)
        
        categorical_applied = False
        try:
            debug_info = model.debug_info()
            model_cats = debug_info.get('pandas_categorical', [])
            if model_cats and len(model_cats) >= 2:
                 if '枠番' in X.columns: X['枠番'] = pd.Categorical(X['枠番'], categories=model_cats[0])
                 if '馬番' in X.columns: X['馬番'] = pd.Categorical(X['馬番'], categories=model_cats[1])
                 categorical_applied = True
        except Exception as e:
            print(f"Error applying categorical definitions: {e}")
        
        if not categorical_applied:
            if '枠番' in X.columns: X['枠番'] = X['枠番'].astype(int)
            if '馬番' in X.columns: X['馬番'] = X['馬番'].astype(int)
            
        try:
            probs = model.predict(X)
        except Exception as e:
            raise e
        
        explanations_list = []
        # SHAP calculation omitted for brevity
        
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            odds = row.get('odds', 0.0)
            popularity = row.get('popularity', 0)
            if odds == 0.0 and '単勝' in row: odds = row.get('単勝', 0.0)
            if popularity == 0 and '人気' in row: popularity = row.get('人気', 0)
            if pd.isna(odds): odds = 0.0
            if pd.isna(popularity): popularity = 0
            
            reasoning = explanations_list[i] if i < len(explanations_list) else {'positive': [], 'negative': []}
            item = row.to_dict()
            if 'horse_name' not in df.columns:
                df['horse_name'] = df['馬名'] if '馬名' in df.columns else df.index.map(lambda x: f"馬{x}")
            
            for k, v in item.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    item[k] = None
                    
            item.update({
                'horse_number': int(row.get('馬番', i + 1)),
                'horse_name': str(row.get('馬名', f'馬{i+1}')),
                'probability': float(probs[i]),
                'odds': float(odds),
                'popularity': int(popularity),
                'expected_value': float(probs[i] * float(odds)),
                'reasoning': reasoning
            })
            results.append(item)
        
        # LTRの場合は「確率」が「実力スコア」になるため、表示上の工夫
        # スコアの最大値で正規化して0-1の範囲に（表示用）
        if model_type == 'ltr':
            max_p = max([r['probability'] for r in results]) if results else 1.0
            min_p = min([r['probability'] for r in results]) if results else 0.0
            range_p = max_p - min_p if max_p != min_p else 1.0
            for r in results:
                # 順位付けに使う生スコアは維持しつつ、表示用にスケーリング
                r['display_probability'] = (r['probability'] - min_p) / range_p
        else:
            for r in results:
                r['display_probability'] = r['probability']

        results.sort(key=lambda x: x['probability'], reverse=True)
        for rank, res in enumerate(results, 1):
            res['predicted_rank'] = rank
            score = res['display_probability']
            ev = res['expected_value']
            # 自信度評価 (LTRとLGBMで基準を変えるか検討)
            if ev >= 1.0 and score >= 0.4:
                res['strategy_decision'] = 'BUY'
                res['analysis'] = {'type': 'recommended', 'message': '★ 推奨馬'}
            else:
                res['strategy_decision'] = 'PASS'
                if score >= 0.4: res['analysis'] = {'type': 'info', 'message': '好走期待'}
                elif ev >= 1.0: res['analysis'] = {'type': 'info', 'message': '期待値あり'}
                else: res['analysis'] = {'type': 'normal', 'message': ''}

        race_name = df.attrs.get('race_name', race_name_default)
        race_data01 = df.attrs.get('race_data01', '')
        race_data02 = df.attrs.get('race_data02', '')

        if race_id:
            print(f"Caching results for cache_key: {cache_key}")
            PREDICTION_CACHE[cache_key] = {
                'df': df,
                'results': results,
                'race_name': race_name,
                'race_data01': race_data01,
                'race_data02': race_data02,
                'timestamp': datetime.now()
            }

    # ここからは、キャッシュの有無に関わらず実行（予算や戦略が変わる可能性があるため）
    recommendations = []
    odds_warning = None
    if race_id and budget > 0:
        try:
             df_preds_alloc = pd.DataFrame(results)
             odds_data = None
             try:
                 odds_data = Odds.scrape(race_id)
             except Exception as oe:
                 print(f"Failed to fetch detailed odds data: {oe}")
             recommendations = BettingAllocator.allocate_budget(df_preds_alloc, budget, odds_data=odds_data, strategy=strategy)
             if not recommendations:
                 odds_warning = "推奨条件を満たす組み合わせが見つかりませんでした (予算不足または確度不足)"
        except Exception as e:
             import traceback
             print(f"Allocation Error: {e}")
             traceback.print_exc()

    confidence_level = 'D'
    if results:
        # LTRの場合は display_probability を使用
        top_prob = results[0]['display_probability']
        top_ev = results[0]['expected_value']
        if top_prob >= 0.5 or top_ev >= 1.5: confidence_level = 'S'
        elif top_prob >= 0.4 or top_ev >= 1.2: confidence_level = 'A'
        elif top_prob >= 0.3 or top_ev >= 1.0: confidence_level = 'B'
        elif top_prob >= 0.2: confidence_level = 'C'
        else: confidence_level = 'D'

    # フロントエンドでの表示確率を display_probability に差し替え
    for r in results:
        r['probability'] = r['display_probability']

    return jsonify({
        'success': True,
        'race_id': race_id,
        'predictions': results,
        'recommendations': recommendations,
        'confidence_level': confidence_level,
        'odds_warning': odds_warning,
        'race_name': race_name,
        'race_info': race_info_default,
        'race_data01': race_data01,
        'race_data02': race_data02,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/demo', methods=['GET'])
def demo():
    """デモデータで予測"""
    model = get_model()
    if model is None:
        return jsonify({'error': 'モデルの読み込みに失敗しました'}), 500
    
    try:
        # サンプルレースデータ
        sample_horses = [
            {'馬番': 1, '馬名': 'ディープインパクト', '単勝': 2.5, '人気': 2, '斤量': 57.0, '年齢': 4, '体重': 480, '体重変化': 0, 'course_len': 2400, 'avg_rank': 2.5, 'win_rate': 0.35, 'place_rate': 0.65, 'race_count': 15, 'jockey_avg_rank': 4.5, 'jockey_win_rate': 0.15, '性': 0, 'race_type': 0},
            {'馬番': 2, '馬名': 'オルフェーヴル', '単勝': 5.8, '人気': 4, '斤量': 57.0, '年齢': 5, '体重': 500, '体重変化': -4, 'course_len': 2400, 'avg_rank': 3.8, 'win_rate': 0.22, 'place_rate': 0.48, 'race_count': 22, 'jockey_avg_rank': 5.2, 'jockey_win_rate': 0.12, '性': 0, 'race_type': 0},
            {'馬番': 3, '馬名': 'キタサンブラック', '単勝': 8.2, '人気': 5, '斤量': 57.0, '年齢': 5, '体重': 520, '体重変化': 2, 'course_len': 2400, 'avg_rank': 4.2, 'win_rate': 0.18, 'place_rate': 0.42, 'race_count': 28, 'jockey_avg_rank': 4.8, 'jockey_win_rate': 0.13, '性': 0, 'race_type': 0},
            {'馬番': 4, '馬名': 'アーモンドアイ', '単勝': 3.2, '人気': 3, '斤量': 55.0, '年齢': 4, '体重': 450, '体重変化': 0, 'course_len': 2400, 'avg_rank': 2.8, 'win_rate': 0.30, 'place_rate': 0.58, 'race_count': 18, 'jockey_avg_rank': 4.2, 'jockey_win_rate': 0.18, '性': 1, 'race_type': 0},
            {'馬番': 5, '馬名': 'コントレイル', '単勝': 4.5, '人気': 4, '斤量': 57.0, '年齢': 3, '体重': 470, '体重変化': 4, 'course_len': 2400, 'avg_rank': 3.0, 'win_rate': 0.28, 'place_rate': 0.52, 'race_count': 12, 'jockey_avg_rank': 5.0, 'jockey_win_rate': 0.11, '性': 0, 'race_type': 0},
            {'馬番': 6, '馬名': 'イクイノックス', '単勝': 1.8, '人気': 1, '斤量': 58.0, '年齢': 4, '体重': 490, '体重変化': -2, 'course_len': 2400, 'avg_rank': 2.2, 'win_rate': 0.40, 'place_rate': 0.72, 'race_count': 10, 'jockey_avg_rank': 4.0, 'jockey_win_rate': 0.20, '性': 0, 'race_type': 0},
            {'馬番': 7, '馬名': 'リバティアイランド', '単勝': 12.0, '人気': 6, '斤量': 54.0, '年齢': 3, '体重': 440, '体重変化': 0, 'course_len': 2400, 'avg_rank': 4.5, 'win_rate': 0.15, 'place_rate': 0.38, 'race_count': 8, 'jockey_avg_rank': 5.5, 'jockey_win_rate': 0.10, '性': 1, 'race_type': 0},
            {'馬番': 8, '馬名': 'ドゥラメンテ', '単勝': 15.0, '人気': 7, '斤量': 57.0, '年齢': 5, '体重': 510, '体重変化': 0, 'course_len': 2400, 'avg_rank': 5.0, 'win_rate': 0.12, 'place_rate': 0.30, 'race_count': 25, 'jockey_avg_rank': 5.8, 'jockey_win_rate': 0.09, '性': 0, 'race_type': 0},
        ]
        
        df = pd.DataFrame(sample_horses)
        return run_prediction_logic(df, 'デモレース - 日本ダービー（G1）', '芝2400m / 良')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """特徴量重要度を取得"""
    model_type = request.args.get('model_type', 'lgbm')
    print(f"API CALL: /api/feature_importance triggered for {model_type}")
    model = get_model(model_type)
    
    if model is None:
        return jsonify({
            'success': False,
            'available': False,
            'message': 'モデルがロードされていません'
        })
    
    try:
        importance = model.get_feature_importance(15)
        
        # Check if importance is available (sum > 0)
        total_importance = importance['importance'].sum()
        is_available = bool(total_importance > 0)
        
        if is_available:
            # Cast to native python types for JSON serialization
            features_data = []
            for _, row in importance.iterrows():
                features_data.append({
                    'feature': str(row['feature']),
                    'importance': float(row['importance'])
                })
        else:
            features_data = []
        
        return jsonify({
            'success': True,
            'features': features_data,
            'available': is_available,
            'message': 'このモデルでは特徴量重要度は利用できません' if not is_available else ''
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """モデル情報を取得"""
    model_type = request.args.get('model_type', 'lgbm')
    print(f"API CALL: /api/model_info triggered for {model_type}")
    model = get_model(model_type)
    
    # モデルファイルのパスを特定
    if model_type == 'ltr':
        model_path = os.path.join(MODEL_DIR, 'standalone_ranking', 'ranking_model.pkl')
    else:
        model_path = os.path.join(MODEL_DIR, 'historical_2010_2026', 'model.pkl')
    
    last_updated = "-"
    if os.path.exists(model_path):
        try:
            mtime = os.path.getmtime(model_path)
            dt = datetime.fromtimestamp(mtime)
            last_updated = dt.strftime('%Y/%m/%d %H:%M')
        except:
            pass

    if model is None:
        return jsonify({
            'success': False,
            'error': 'モデルがロードされていません',
            'last_updated': last_updated
        })
    
    try:
        algo_map = {
            'lgbm': 'LightGBM (Historical)',
            'rf': 'Random Forest',
            'ltr': 'LambdaMART (Ranking)',
            'pytorch_mlp': 'PyTorch MLP',
            'catboost': 'CatBoost',
            'xgb': 'XGBoost',
            'gbc': 'Gradient Boosting'
        }
        
        if hasattr(model, 'models') and hasattr(model, 'weights'): # Ensemble check
            algo_name = 'Ensemble (LGBM + RF)'
        else:
            raw_type = getattr(model, 'model_type', 'unknown')
            algo_name = algo_map.get(raw_type, raw_type)
            
        feature_count = len(model.feature_names) if model.feature_names else 0
        
        # モデルごとの指標 (LTRの場合は2025年シミュレーション結果を表示)
        metrics = {'auc': 0.802, 'recovery_rate': 114.1}
        if model_type == 'ltr':
            metrics = {'auc': 0.805, 'recovery_rate': 301.2} # LTR Anchor
        
        return jsonify({
            'success': True,
            'algorithm': str(algo_name),
            'target': '複勝（3着以内）',
            'source': 'netkeiba.com',
            'feature_count': int(feature_count),
            'last_updated': last_updated,
            'metrics': metrics
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# IPAT連携 API (Selenium Browser Automation - Direct IPAT Access)
# ==============================================================================
# from modules.ipat_connector import IpatConnector, IpatBetItem # REMOVED: HTTP request is deprecated
# from modules.netkeiba_automator import NetkeibaAutomator # DEPRECATED: Now using direct IPAT access
from modules.ipat_direct_automator import IpatDirectAutomator


def convert_recommendations_to_bets(recommendations: list) -> list:
    """
    買い目推奨データをIPAT vote()用のフォーマットに変換する
    
    Args:
        recommendations: BettingAllocatorの出力リスト
        
    Returns:
        list: vote()メソッド用のbetsリスト
    """
    bets = []
    
    for rec in recommendations:
        # method変換: SINGLE -> '通常', BOX -> 'ボックス', FORMATION -> 'フォーメーション'
        # BettingAllocatorの出力形式: method='BOX', horse_numbers=[1, 2, 3]
        # またはmethod='FORMATION', formation_horses=[[12,9], [12,9,4], [12,9,4,13]]
        method_raw = rec.get('method', 'SINGLE')
        
        # IPAT用のmethod名
        if method_raw in ['SINGLE', '通常']:
            method = '通常'
        elif method_raw in ['BOX', 'ボックス']:
            method = 'ボックス'
        elif method_raw in ['FORMATION', 'フォーメーション']:
            method = 'フォーメーション'
        elif method_raw in ['NAGASHI', '流し', '2軸流し']:
            # 流しはフォーメーションに変換して処理する
            method = 'フォーメーション'
        else:
            method = '通常'  # デフォルト
        
        # BettingAllocatorは'bet_type'を使用するが、後方互換性のため'type'もサポート
        bet_type = rec.get('bet_type') or rec.get('type')
        
        # フォーメーションの場合は formation_horses を優先
        # formation_horses: [[1着馬], [2着馬], [3着馬]] の形式
        if method == 'フォーメーション':
            horses = rec.get('formation_horses') or rec.get('formation') or rec.get('horses')
            
            # NAGASHI -> Formation 変換
            # 流しの場合は axis/partners 構造をフォーメーション形式に変換
            if method_raw in ['NAGASHI', '流し', '2軸流し']:
                nagashi_horses = rec.get('nagashi_horses') or rec.get('formation') or rec.get('horses')
                axis = rec.get('axis') or rec.get('axis_horses', [])
                partners = rec.get('partners') or rec.get('partner_horses', [])
                
                # dict形式の場合
                if isinstance(nagashi_horses, dict):
                    axis = nagashi_horses.get('axis', axis)
                    partners = nagashi_horses.get('partners', partners)
                # リスト形式の場合 (BettingAllocatorの出力など)
                elif isinstance(nagashi_horses, list) and len(nagashi_horses) >= 2:
                    axis = nagashi_horses[0]
                    partners = nagashi_horses[1]
                
                # リスト形式に正規化
                if not isinstance(axis, list):
                    axis = [axis] if axis is not None else []
                if not isinstance(partners, list):
                    partners = [partners] if partners is not None else []
                
                # 流し -> フォーメーション変換
                # 3連複 軸1頭流し: [[軸], [相手], [相手]]
                # 3連複 2軸流し: [[軸1], [軸2], [相手]]
                # 3連単 軸1頭1着固定: [[軸], [相手], [相手]]
                # 馬連/ワイド/馬単: [[軸], [相手]]
                
                if bet_type == '3連複' and method_raw == '2軸流し':
                    # 2軸流しの入力[ [a1, a2], [p1, p2...] ] を分解
                    if len(axis) >= 2:
                        horses = [[axis[0]], [axis[1]], partners]
                    else:
                        horses = [axis, partners, partners]
                    print(f"Converted 2-axis NAGASHI to Formation: {horses}")
                elif bet_type in ['3連複', '3連単']:
                    # 軸1頭流し: 軸 + 相手から2頭
                    horses = [axis, partners, partners]
                    print(f"Converted NAGASHI to Formation: {horses}")
                elif bet_type in ['馬連', '馬単', 'ワイド']:
                    horses = [axis, partners]
                    print(f"Converted NAGASHI to Formation: {horses}")
                else:
                    horses = [axis, partners]
        else:
            # 'horse_numbers'(BettingAllocator)と'horses'(旧形式)の両方をサポート
            horses = rec.get('horse_numbers') or rec.get('horses') or rec.get('formation')
        
        # 金額決定
        if method == '通常':
            amount = rec.get('total_amount', 100)
        else:
            # BOX/フォーメーションの場合は1点あたりの金額
            amount = rec.get('unit_amount', 100)
            
        bet = {
            'type': bet_type,
            'horses': horses,
            'amount': amount,
            'method': method
        }
        
        bets.append(bet)
    
    return bets



def load_ipat_credentials():
    """
    IPAT認証情報を読み込む
    
    優先順位:
    1. scripts/debug/ipat_secrets.json (あれば)
    2. 環境変数 (フォールバック)
    
    Returns:
        tuple: (inetid, subscriber_no, pin, pars_no)
    """
    # JSONファイルから読み込み試行
    secrets_path = os.path.join(os.path.dirname(__file__), 'scripts', 'debug', 'ipat_secrets.json')
    
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = json.load(f)
                print(f"IPAT認証情報を読み込みました: {secrets_path}")
                return (
                    secrets.get('inetid', ''),
                    secrets.get('subscriber_no', ''),
                    secrets.get('pin', ''),
                    secrets.get('pars_no', '')
                )
        except Exception as e:
            print(f"JSONファイルの読み込みに失敗: {e}")
            # フォールバックで環境変数を使用
    
    # 環境変数から読み込み
    print("環境変数からIPAT認証情報を読み込みます")
    return (
        os.environ.get('IPAT_INETID', ''),
        os.environ.get('IPAT_SUBSCRIBER_NO', ''),
        os.environ.get('IPAT_PIN', ''),
        os.environ.get('IPAT_PARS_NO', '')
    )


# IPAT自動化インスタンスをグローバル保持 (セッション維持のため)
ipat_automator = None

@app.route('/api/ipat/launch_browser', methods=['POST'])
def launch_ipat_browser():
    """
    Seleniumでブラウザを起動し、IPAT投票画面に直接アクセスして買い目を入力する
    """
    global ipat_automator
    
    try:
        data = request.json
        race_id = data.get('race_id')
        recommendations = data.get('recommendations', [])  # フロントエンドから推奨データを受け取る
        
        if not race_id:
            return jsonify({'success': False, 'error': 'レースIDが必要です'}), 400
        
        if not recommendations:
            return jsonify({'success': False, 'error': '買い目データが必要です'}), 400
        
        # 推奨データをbets形式に変換
        bets = convert_recommendations_to_bets(recommendations)
        
        # 認証情報を取得 (JSONファイル優先、環境変数フォールバック)
        inetid, subscriber_no, pin, pars_no = load_ipat_credentials()
        
        # 認証情報チェック
        if not all([subscriber_no, pin, pars_no]):
            return jsonify({
                'success': False, 
                'error': 'IPAT認証情報が設定されていません。scripts/debug/ipat_secrets.json または環境変数を設定してください。'
            }), 400
            
        print(f"Launching IPAT for Race {race_id}, Bets: {len(bets)}")
        print(f"Converted bets: {bets}")
        
        # IPAT自動化インスタンスの管理
        # 既存のインスタンスがあり、かつドライバが生存しているかチェック
        is_active = False
        if ipat_automator is not None:
            try:
                # ドライバが生きているか確認 (タイトル取得などで)
                _ = ipat_automator.driver.title
                is_active = True
                print("Existing IPAT automator instance found and active.")
            except:
                print("Existing IPAT automator instance found but driver is dead.")
                ipat_automator = None
        
        if not is_active:
            print("Creating new IPAT automator instance...")
            # 本番用にdebug_mode=Falseで高速化
            ipat_automator = IpatDirectAutomator(debug_mode=False)
        
        # 1. ログイン (既存セッションがある場合は内部でスキップされる)
        # ログイン処理は毎回呼び出すが、クラス内部で「既にログイン済み」ならスキップする実装になっている
        login_success, login_msg = ipat_automator.login(inetid, subscriber_no, pin, pars_no)
        
        if not login_success:
            # ログイン失敗時はインスタンスをリセットした方が安全かも
            # ipat_automator.close() # 失敗理由によるが、閉じた方が無難
            return jsonify({
                'success': False, 
                'error': f'IPATログインに失敗しました: {login_msg}'
            }), 400
        
        # 2. 投票実行 (確認画面で停止)
        vote_success, vote_msg = ipat_automator.vote(race_id, bets, stop_at_confirmation=True)
        
        if vote_success:
            return jsonify({
                'success': True, 
                'message': f'{vote_msg}\n\n✅ 合計金額は自動計算・入力済みです（Extreme Speed Mode）。\n⚠️ 内容を確認し、「入力終了」→「投票」ボタンを手動で押してください。'
            })
        else:
            # 投票失敗時はブラウザを閉じる
            ipat_automator.close()
            ipat_automator = None # インスタンスもリセット
            return jsonify({
                'success': False, 
                'error': f'投票処理に失敗しました: {vote_msg}'
            }), 500
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # エラー発生時はブラウザを閉じる
        if ipat_automator:
            ipat_automator.close()
            ipat_automator = None # インスタンスもリセット
        return jsonify({'success': False, 'error': f'システムエラー: {str(e)}'}), 500



if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
