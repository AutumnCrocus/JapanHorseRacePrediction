"""
Walk-Forward Validation Implementations
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .constants import MODEL_DIR
from .training import HorseRaceModel, EnsembleModel
from .preprocessing import prepare_training_data

class WalkForwardValidator:
    """時系列検証クラス"""
    
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.results = None
        self.horse_results = None
        self.peds = None
        
    def load_data(self, results_path: str, horse_results_path: str, peds_path: str):
        """データ読み込み"""
        if os.path.exists(results_path):
            self.results = pd.read_pickle(results_path)
        if os.path.exists(horse_results_path):
            self.horse_results = pd.read_pickle(horse_results_path)
        if os.path.exists(peds_path):
            self.peds = pd.read_pickle(peds_path)
            
    def run_validation(self, start_year: int, end_year: int):
        """
        Walk-Forward Validationを実行
        
        Args:
            start_year: 検証開始年 (例: 2021 なら 2016-2020学習 -> 2021検証)
            end_year: 検証終了年
        """
        if self.results is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        print(f"=== Walk-Forward Validation ({start_year} - {end_year}) ===")
        
        metrics_history = []
        
        for year in range(start_year, end_year + 1):
            train_end_year = year - 1
            print(f"\nTarget Year: {year}")
            print(f"Training Period: ~ {train_end_year}")
            
            # データ分割
            # レースID (YYYY...) でフィルタリング
            train_mask = self.results.index.astype(str).str[:4].astype(int) <= train_end_year
            test_mask = self.results.index.astype(str).str[:4].astype(int) == year
            
            train_results = self.results[train_mask]
            test_results = self.results[test_mask]
            
            if test_results.empty:
                print(f"Warning: No data for {year}. Skipping.")
                continue
                
            print(f"Train samples: {len(train_results)}, Test samples: {len(test_results)}")
            
            # 前処理と学習
            # 注意: 毎年Processorを作り直す（リーク防止）
            X_train, y_train, processor, engineer = prepare_training_data(
                train_results, self.horse_results, self.peds, scale=False # LightGBM/RF main
            )
            
            # モデル学習（簡易版としてLGBM単体、またはアンサンブル）
            # 時間短縮のためLGBMのみとするか、ユーザー要望通りしっかりやるか
            # ここではLGBMを使用
            model = HorseRaceModel(model_type='lgbm')
            model.train(X_train, y_train)
            
            # テストデータ前処理 (学習済みProcessorを使用)
            # テストデータに対しても特徴量作成パイプラインを適用
            # prepare_training_dataではなく、手動で適用する必要がある（Processor再利用のため）
            # ここはevaluation.py単体で完結させるため、少し冗長だがロジックを書くか、
            # main.pyのevalロジックを参考にする
            
            # Processor/Engineerを使ってテストデータ作成
            test_df = processor.process_results(test_results)
            
            # 特徴量追加 (horse_resultsなどは全期間渡してOK -> 時系列処理が実装済みだから)
            if self.horse_results is not None:
                hr_tmp = self.horse_results.copy()
                hr_tmp.columns = hr_tmp.columns.str.replace(' ', '')
                if '着順' in hr_tmp.columns:
                    hr_tmp['着順'] = pd.to_numeric(hr_tmp['着順'], errors='coerce')
                
                test_df = engineer.add_horse_history_features(test_df, hr_tmp)
                test_df = engineer.add_course_suitability_features(test_df, hr_tmp)
            
            test_df = engineer.add_jockey_features(test_df)
            
            if self.peds is not None:
                test_df = engineer.add_pedigree_features(test_df, self.peds)
                
            test_df = engineer.create_target(test_df, target_type='place')
            
            # カテゴリエンコード
            cat_cols = ['性', 'race_type', 'weather', 'ground_state', 'sire', 'dam']
            cat_cols = [c for c in cat_cols if c in test_df.columns]
            test_df = processor.encode_categorical(test_df, cat_cols)
            
            # 特徴量選択
            features = [c for c in model.feature_names if c in test_df.columns]
            for c in model.feature_names:
                if c not in test_df.columns: test_df[c] = 0
            
            X_test = test_df[model.feature_names].fillna(0) # 簡易補完
            if hasattr(processor, 'scaler') and processor.scaler:
                X_test = processor.transform_scale(X_test)
                
            y_test = test_df['target']
            
            # 予測と評価
            y_pred = (model.predict(X_test) >= 0.5).astype(int)
            y_proba = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            print(f"Result {year}: AUC={auc:.4f}, Accuracy={acc:.4f}")
            
            metrics = {
                'year': year,
                'auc': auc,
                'accuracy': acc,
                'f1': f1,
                'train_size': len(train_results),
                'test_size': len(test_results)
            }
            metrics_history.append(metrics)
            
        return pd.DataFrame(metrics_history)
