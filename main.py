"""
競馬予想AI - メインスクリプト
データ収集から予測までの一連のパイプラインを実行
"""

import os
import pickle
import argparse
import pandas as pd
from datetime import datetime

from modules.constants import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
    RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE, RETURN_FILE
)
from modules.scraping import Results, HorseResults, Peds, Return, get_race_id_list, update_data
from modules.preprocessing import DataProcessor, FeatureEngineer, prepare_training_data
from modules.training import HorseRaceModel, RacePredictor, create_sample_model, EnsembleModel
from modules.simulation import BettingSimulator, run_simulation_report
from modules.evaluation import WalkForwardValidator


def ensure_directories():
    """必要なディレクトリを作成"""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)


def save_data(data: pd.DataFrame, filepath: str):
    """データを保存"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"データを保存しました: {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """データを読み込み"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return pd.DataFrame()


def scrape_data(start_year: int, end_year: int = None, place_codes: list = None, sample: bool = False):
    """
    データをスクレイピング
    
    Args:
        start_year: 開始年
        end_year: 終了年
        place_codes: 競馬場コード
        sample: サンプルモードかどうか
    """
    ensure_directories()
    
    if end_year is None:
        end_year = start_year
        
    print(f"=== {start_year}年〜{end_year}年のデータをスクレイピング ===")
    
    # レースIDリストを生成
    race_id_list = get_race_id_list(start_year, end_year, place_codes)
    
    limit = None
    if sample:
        # サンプルモード: 上限50レース
        limit = 50
        print(f"サンプルモード: 最大{limit}レースを取得します（成功するまで続行）")
    
    # レース結果をスクレイピング
    print("\n[1/4] レース結果の取得...")
    results = Results.scrape(race_id_list, limit=limit)
    if not results.empty:
        # 既存データと統合
        old_results = load_data(os.path.join(RAW_DATA_DIR, RESULTS_FILE))
        results = update_data(old_results, results)
        save_data(results, os.path.join(RAW_DATA_DIR, RESULTS_FILE))
        
        # 馬IDを抽出
        horse_ids = results['horse_id'].unique().tolist()
        
        if sample:
            horse_ids = horse_ids[:100]
        
        # 馬の過去成績をスクレイピング
        print("\n[2/4] 馬の過去成績の取得...")
        horse_results = HorseResults.scrape(horse_ids)
        if not horse_results.empty:
            old_horse_results = load_data(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
            horse_results = update_data(old_horse_results, horse_results)
            save_data(horse_results, os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
        
        # 血統データをスクレイピング
        print("\n[3/4] 血統データの取得...")
        peds = Peds.scrape(horse_ids)
        if not peds.empty:
            old_peds = load_data(os.path.join(RAW_DATA_DIR, PEDS_FILE))
            peds = update_data(old_peds, peds)
            save_data(peds, os.path.join(RAW_DATA_DIR, PEDS_FILE))
        
        # 払戻データをスクレイピング
        print("\n[4/4] 払戻データの取得...")
        # 取得できたレースIDに対して払戻データを取得
        valid_race_ids = results.index.unique().tolist()
        if sample:
             # サンプルモードならレース結果と同じ数だけ取得（または半分など調整可能）
             pass
        
        returns = Return.scrape(valid_race_ids)
        if not returns.empty:
            old_returns = load_data(os.path.join(RAW_DATA_DIR, RETURN_FILE))
            returns = update_data(old_returns, returns)
            save_data(returns, os.path.join(RAW_DATA_DIR, RETURN_FILE))
    else:
        print("レース結果が見つかりませんでした。対象年またはレースIDを確認してください。")
    
    print("\n=== スクレイピング完了 ===")


def train_model(algo: str = 'lgbm', ensemble: bool = False):
    """モデルを学習"""
    ensure_directories()
    
    if ensemble:
        print("=== モデル学習 (アンサンブル) ===")
def train_model(algo: str = 'lgbm', ensemble: bool = False, full_train: bool = False):
    """
    モデル学習を実行
    
    Args:
        algo: アルゴリズム ('lgbm', 'rf' など)
        ensemble: アンサンブル学習を行うかどうか
        full_train: 全データで学習するかどうか（予測用）
    """
    ensure_directories()
    
    # データ読み込み
    results = load_data(os.path.join(RAW_DATA_DIR, RESULTS_FILE))
    horse_results = load_data(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
    peds = load_data(os.path.join(RAW_DATA_DIR, PEDS_FILE))
    
    if results.empty:
        print("学習データがありません。scrapeコマンドでデータを取得してください。")
        print("サンプルモデルを作成します...")
        model = create_sample_model(model_type=algo)
        model.save()
        return model
    
    print(f"学習データ数: {len(results)}")
    
    # ターゲット生成と特徴量エンジニアリング用データの準備
    # アンサンブルの場合はスケーリングありの方が良い場合も
    scale = True if algo in ['pytorch_mlp'] else False
    
    X, y, processor, engineer = prepare_training_data(results, horse_results, peds, scale=scale)
    
    test_size = 0.0 if full_train else 0.2
    
    if ensemble:
        print("\n=== モデル学習 (アンサンブル) ===")
    else:
        print(f"=== モデル学習 ({algo}) ===")
    
    print(f"特徴量数: {len(X.columns)}")
    
    if ensemble:
        print("\n[2/3] アンサンブル学習 (LGBM + Random Forest)...")
        # 簡易的にLGBMとRFのアンサンブルを作成
    # 特徴量重要度
    print("\n特徴量重要度（上位10）:")
    importance = model.get_feature_importance(10)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    # モデル保存
    print("\n[3/3] モデル保存...")
    model.save()
    
    # ProcessorとEngineerも保存
    with open(os.path.join(MODEL_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    with open(os.path.join(MODEL_DIR, 'engineer.pkl'), 'wb') as f:
        pickle.dump(engineer, f)
    
    print("\n=== 学習完了 ===")
    
    return model


def evaluate_model(year: int):
    """
    指定した年のデータでモデルを評価
    
    Args:
        year: 評価対象年
    """
    ensure_directories()
    print(f"=== モデル評価 ({year}年データ) ===")
    
    # データ読み込み
    results = load_data(os.path.join(RAW_DATA_DIR, RESULTS_FILE))
    horse_results = load_data(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE))
    peds = load_data(os.path.join(RAW_DATA_DIR, PEDS_FILE))
    
    if results.empty:
        print("レース結果データがありません。先にスクレイピングを実行してください。")
        return

    # 指定年のデータのみ抽出
    # indexが文字列型のレースID (YYYY...) と仮定
    target_results = results[results.index.astype(str).str.startswith(str(year))]
    
    if target_results.empty:
        print(f"{year}年のデータが見つかりません。")
        print(f"コマンド: python main.py scrape --year {year} を実行してデータを取得してください。")
        return
    
    print(f"{year}年のデータ数: {len(target_results)}レース")

    # モデルとプロセッサの読み込み
    model_path = os.path.join(MODEL_DIR, 'horse_race_model.pkl') # 通常モデル
    ensemble_path = os.path.join(MODEL_DIR, 'model_lgbm_0.pkl') # アンサンブルの一部が存在するか確認
    
    model = None
    if os.path.exists(model_path):
         model = HorseRaceModel()
         model.load(model_path)
         print(f"モデルを読み込みました: {model_path} ({model.model_type})")
    elif os.path.exists(ensemble_path):
        # アンサンブルモデルとして読み込みトライ
        print("アンサンブルモデルを読み込みます...")
        model = EnsembleModel()
        model.load(MODEL_DIR)
    else:
        print("学習済みモデルが見つかりません。先に python main.py train を実行してください。")
        return

    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    if not os.path.exists(processor_path) or not os.path.exists(engineer_path):
         print("前処理オブジェクト(processor/engineer)が見つかりません。")
         return

    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    with open(engineer_path, 'rb') as f:
        engineer = pickle.load(f)

    # データ作成 (prepare_training_dataと同様だが、学習済みprocessor/engineerを使う)
    print("\n[1/3] データ前処理...")
    
    # 手動でパイプライン実行 (prepare_training_dataは新規作成してしまうため)
    df = processor.process_results(target_results)
    
    # 特徴量追加
    if not horse_results.empty:
        # カラム名正規化など
        horse_results_tmp = horse_results.copy()
        horse_results_tmp.columns = horse_results_tmp.columns.str.replace(' ', '')
        if '着順' in horse_results_tmp.columns:
            horse_results_tmp['着順'] = pd.to_numeric(horse_results_tmp['着順'], errors='coerce')
            
        df = engineer.add_horse_history_features(df, horse_results_tmp)
        df = engineer.add_course_suitability_features(df, horse_results_tmp)
    
    df = engineer.add_jockey_features(df)
    
    if not peds.empty:
        df = engineer.add_pedigree_features(df, peds)
        
    df = engineer.create_target(df, target_type='place') # 評価用ターゲット
    
    # カテゴリエンコード (学習時のEncoderを使用)
    categorical_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    # DataProcessor.encode_categorical は新規fitしてしまう可能性があるため注意が必要だが、
    # 実装を見ると if col not in self.label_encoders: で新規作成、elseで transform しているので
    # pickleロードしたprocessorなら大丈夫。
    df = processor.encode_categorical(df, categorical_cols)

    # 特徴量カラム選択
    feature_cols = [c for c in model.feature_names if c in df.columns]
    
    # 足りないカラムは0埋め
    for col in model.feature_names:
        if col not in df.columns:
            df[col] = 0
            
    X = df[model.feature_names].copy()
    X = X.fillna(X.median())
    
    if hasattr(processor, 'scaler') and processor.scaler:
         X = processor.transform_scale(X)
    y = df['target'].copy()

    print(f"評価データ数: {len(X)}")

    # 予測
    print("\n[2/3] 予測実行...")
    # predict method handles probing for predict_proba vs predict automatically in our wrapper?
    # Actually our HorseRaceModel.predict returns probability for RF/LGBM/etc. if implemented that way.
    # Let's check HorseRaceModel.predict.
    # It returns probabilities for some, but let's be sure.
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 評価
    print("\n[3/3] 評価結果:")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except:
        auc = 0.0

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")
    
    # 簡易シミュレーション (回収率)
    # 単勝/複勝データがあれば計算可能だが、preprocessingで捨てられている可能性も。
    # raw dataの '単勝' カラムなどが df に残っているか確認。
    # process_resultsで df=results_df.copy() なので残っているはず。
    
    if '単勝' in df.columns and '着順' in df.columns:
        print("\n--- 回収率シミュレーション (Threshold 0.5で複勝と仮定) ---")
        # target_type='place' なので yは3着以内。
        # 実際にはBET判定は y_pred (prob >= 0.5)
        
        # 複勝オッズはここにはないかもしれない（Returnテーブルが必要）
        # なので、単勝で1着狙いの場合のシミュレーションだけ行うか、
        # あるいは「的中率」だけ出す。
        
        # もし複勝オッズがあれば良いが、standard scraping logic usually gets basic info.
        # Let's check Return scraping.
        
        # 簡易的に単勝回収率 (閾値を厳しくして 1着予測)
        # model output is 'place' (top 3) probability usually if trained on place target?
        # create_target(..., target_type='place') is used in train flow?
        # In prepare_training_data: df = engineer.create_target(df, target_type='place')
        # So the model predicts probability of being in top 3.
        
        # Let's stick to Metrics for now. Recover rates need accurate Return table data linked.
        pass


def predict_race(race_data: pd.DataFrame = None, file_path: str = None) -> pd.DataFrame:
    """レースの予測を行う"""
    ensure_directories()
    
    # モデル読み込み
    model = HorseRaceModel()
    model.load()
    
    # ProcessorとEngineerを読み込み
    processor = None
    engineer = None
    
    processor_path = os.path.join(MODEL_DIR, 'processor.pkl')
    engineer_path = os.path.join(MODEL_DIR, 'engineer.pkl')
    
    if os.path.exists(processor_path):
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
    
    if os.path.exists(engineer_path):
        with open(engineer_path, 'rb') as f:
            engineer = pickle.load(f)
    
    predictor = RacePredictor(model, processor, engineer)
    
    if file_path:
        print(f"データを読み込んでいます: {file_path}")
        race_data = pd.read_csv(file_path)
    elif race_data is None:
        # デモ用のサンプルデータ
        race_data = pd.DataFrame({
            '馬番': [1, 2, 3, 4, 5],
            '馬名': ['馬A', '馬B', '馬C', '馬D', '馬E'],
            '単勝': [3.5, 8.2, 12.0, 5.5, 25.0],
            '人気': [1, 3, 4, 2, 5]
        })
    
    predictions = predictor.predict_race(race_data)
    
    return predictions


def run_demo():
    """デモモードで実行"""
    print("=== 競馬予想AI デモモード ===\n")
    
    # サンプルモデルを作成
    print("[1/3] サンプルモデルの作成...")
    model = create_sample_model()
    
    # モデルを保存
    ensure_directories()
    model.save()
    
    # デモ予測
    print("\n[2/3] サンプルレースの予測...")
    import numpy as np
    np.random.seed(123)
    
    sample_race = pd.DataFrame({
        '枠番': [1, 2, 3, 4, 5, 6, 7, 8],
        '馬番': [1, 2, 3, 4, 5, 6, 7, 8],
        '馬名': ['ディープインパクト', 'オルフェーヴル', 'キタサンブラック', 
                 'アーモンドアイ', 'コントレイル', 'イクイノックス', 
                 'リバティアイランド', 'ドゥラメンテ'],
        '斤量': [57.0, 57.0, 57.0, 55.0, 57.0, 58.0, 54.0, 57.0],
        '単勝': [2.5, 5.8, 8.2, 3.2, 4.5, 1.8, 12.0, 15.0],
        '人気': [2, 4, 5, 3, 4, 1, 6, 7],
        '年齢': [4, 5, 5, 4, 3, 4, 3, 5],
        '体重': [480, 500, 520, 450, 470, 490, 440, 510],
        '体重変化': [0, -4, 2, 0, 4, -2, 0, 0],
        'course_len': [2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400],
        'avg_rank': [2.5, 3.8, 4.2, 2.8, 3.0, 2.2, 4.5, 5.0],
        'win_rate': [0.35, 0.22, 0.18, 0.30, 0.28, 0.40, 0.15, 0.12],
        'place_rate': [0.65, 0.48, 0.42, 0.58, 0.52, 0.72, 0.38, 0.30],
        'race_count': [15, 22, 28, 18, 12, 10, 8, 25],
        'jockey_avg_rank': [4.5, 5.2, 4.8, 4.2, 5.0, 4.0, 5.5, 5.8],
        'jockey_win_rate': [0.15, 0.12, 0.13, 0.18, 0.11, 0.20, 0.10, 0.09],
        '性': [0, 0, 0, 1, 0, 0, 1, 0],
        'race_type': [0, 0, 0, 0, 0, 0, 0, 0]
    })
    
    X_demo = sample_race[model.feature_names]
    probs = model.predict(X_demo)
    
    sample_race['予測確率'] = probs
    sample_race['予測順位'] = sample_race['予測確率'].rank(ascending=False).astype(int)
    sample_race = sample_race.sort_values('予測順位')
    
    print("\n予測結果:")
    print("-" * 60)
    for _, row in sample_race.iterrows():
        print(f"  {row['予測順位']:2d}位: {row['馬名']} "
              f"(確率: {row['予測確率']:.1%}, 単勝: {row['単勝']:.1f}倍)")
    
    # シミュレーション
    print("\n[3/3] 回収率シミュレーション...")
    sample_race['着順'] = [3, 5, 6, 2, 4, 1, 7, 8]  # 仮の着順
    sample_race.index = ['demo_race'] * len(sample_race)
    
    simulator = BettingSimulator(initial_balance=10000)
    result = simulator.simulate(sample_race, None, bet_type='place', threshold=0.4)
    
    print("\nシミュレーション結果:")
    print(f"  初期資金: {result['initial_balance']:,}円")
    print(f"  最終資金: {result['final_balance']:,}円")
    print(f"  賭け回数: {result['total_bets']}回")
    print(f"  的中率: {result['win_rate']:.1f}%")
    print(f"  回収率: {result['recovery_rate']:.1f}%")
    
    print("\n=== デモ完了 ===")


def run_backtest(start_year: int, end_year: int):
    """
    バックテスト（時系列検証）を実行
    
    Args:
        start_year: 検証開始年
        end_year: 検証終了年
    """
    ensure_directories()
    
    validator = WalkForwardValidator(RAW_DATA_DIR)
    
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    horse_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    print(f"Loading data from {RAW_DATA_DIR}...")
    validator.load_data(results_path, horse_path, peds_path)
    
    metrics_df = validator.run_validation(start_year, end_year)
    
    print(f"\n=== Validation Summary ({start_year}-{end_year}) ===")
    print(metrics_df)
    
    # 結果を保存
    output_path = os.path.join(PROCESSED_DATA_DIR, f'backtest_metrics_{start_year}_{end_year}.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='競馬予想AI')
    parser.add_argument('command', choices=['scrape', 'train', 'predict', 'demo', 'evaluate', 'backtest'],
                        help='実行するコマンド')
    parser.add_argument('--year', type=int, help='対象年 (単年指定用)')
    parser.add_argument('--start-year', type=int, help='開始年')
    parser.add_argument('--end-year', type=int, help='終了年')
    parser.add_argument('--sample', action='store_true', help='サンプルモード')
    parser.add_argument('--input', type=str, help='予測用入力CSVファイル')
    parser.add_argument('--algo', type=str, default='lgbm', choices=['lgbm', 'rf', 'gbc', 'xgb', 'catboost', 'pytorch_mlp'], help='学習アルゴリズム')
    parser.add_argument('--ensemble', action='store_true', help='アンサンブル学習を行う')
    
    parser.add_argument('--full-train', action='store_true', help='全データで学習 (予測用)')
    
    args = parser.parse_args()
    
    if args.command == 'scrape':
        # 引数の優先順位: --start-year/--end-year
        start = args.start_year or 2020
        end = args.end_year or 2024
        run_scraping(start, end)
    elif args.command == 'train':
        train_model(algo=args.algo, ensemble=args.ensemble, full_train=args.full_train)
    elif args.command == 'predict':
        predictions = predict_race(file_path=args.input)
        print(predictions)
    elif args.command == 'evaluate':
        year = args.year or 2025
        evaluate_model(year)
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'backtest':
        start = args.start_year or 2021
        end = args.end_year or 2025
        run_backtest(start, end)


if __name__ == '__main__':
    main()
