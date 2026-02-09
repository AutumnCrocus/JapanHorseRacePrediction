"""
機械学習モデルモジュール
LightGBMを使用した競馬予測モデル
"""

import pickle
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .constants import MODEL_DIR

# try:
#     import shap
# except ImportError:
#     shap = None
shap = None

# try:
#     import xgboost as xgb
# except ImportError:
#     xgb = None
xgb = None

# try:
#     import catboost as cb
# except ImportError:
#     cb = None
cb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    optim = None


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.preprocessing import StandardScaler

class SimpleMLP(nn.Module if torch else object):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.output(x)
        return self.sigmoid(x)


class HorseRaceModel:
    """競馬予測モデルクラス (複数アルゴリズム対応)"""
    
    def __init__(self, model_type: str = 'lgbm', model_params: dict = None):
        """
        初期化
        
        Args:
            model_type: モデルの種類 ('lgbm', 'rf', 'gbc', 'xgb', 'catboost')
            model_params: モデルのパラメータ
        """
        self.model_type = model_type
        self.model_params = model_params or self._get_default_params()
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def _get_default_params(self) -> dict:
        """各モデルのデフォルトパラメータを取得"""
        if self.model_type == 'lgbm':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 500,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type == 'rf':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type == 'gbc':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        elif self.model_type == 'xgb':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'random_state': 42
            }
        elif self.model_type == 'catboost':
            return {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'random_seed': 42,
                'verbose': False
            }
        elif self.model_type == 'pytorch_mlp':
            return {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.001,
                'patience': 10
            }
        return {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, early_stopping_rounds: int = 50,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              sample_weight: np.ndarray = None) -> dict:
        """モデルを学習
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
            test_size: 検証データの割合
            early_stopping_rounds: 早期終了ラウンド数
            X_val: 検証用特徴量 (外部指定)
            y_val: 検証用ターゲット (外部指定)
            sample_weight: サンプル重み (時系列重み付け学習用)
        """
        self.feature_names = X.columns.tolist()
        
        # データ分割
        if X_val is not None and y_val is not None:
            # 外部から検証データが提供された場合
            X_train, y_train = X, y
            # X_val, y_val はそのまま使用
        elif test_size > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            # 全データ学習 (検証データも学習データと同じにする)
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # 重みの分割 (sample_weightが指定されている場合)
        w_train = None
        if sample_weight is not None:
            # sample_weight は pd.Series または np.ndarray
            if isinstance(sample_weight, np.ndarray):
                sample_weight = pd.Series(sample_weight, index=X.index)
            
            if X_val is not None and y_val is not None:
                # 外部検証データの場合、重みは訓練データ全体分
                w_train = sample_weight.values
            elif test_size > 0:
                # 内部分割の場合、X_trainのインデックスで抽出
                w_train = sample_weight.loc[X_train.index].values
            else:
                w_train = sample_weight.values
        
        if self.model_type == 'lgbm':
            self._train_lgbm(X_train, y_train, X_val, y_val, early_stopping_rounds, w_train)
        elif self.model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
        elif self.model_type == 'gbc':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
        elif self.model_type == 'xgb':
            if xgb is None: raise ImportError("xgboost is not installed")
            self.model = xgb.XGBClassifier(**self.model_params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                           early_stopping_rounds=early_stopping_rounds, verbose=False)
        elif self.model_type == 'catboost':
            if cb is None: raise ImportError("catboost is not installed")
            self.model = cb.CatBoostClassifier(**self.model_params)
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                           early_stopping_rounds=early_stopping_rounds)
        elif self.model_type == 'pytorch_mlp':
            if torch is None: raise ImportError("torch is not installed")
            self._train_pytorch(X_train, y_train, X_val, y_val)

        # 特徴量重要度の作成
        self._set_feature_importance()
        
        # 検証データで評価
        y_pred_proba = self.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }
        return metrics

    def _train_lgbm(self, X_train, y_train, X_val, y_val, early_stopping_rounds, sample_weight=None):
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = self.model_params.copy()
        n_estimators = params.pop('n_estimators', 500)
        
        self.model = lgb.train(
            params, train_data, num_boost_round=n_estimators,
            valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )

    def _train_pytorch(self, X_train, y_train, X_val, y_val):
        """PyTorchモデルの学習"""
        # データセット作成
        # dataframe to tensor
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.model_params['batch_size'], shuffle=True)
        
        # モデル初期化
        input_dim = X_train.shape[1]
        self.model = SimpleMLP(input_dim)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_params['learning_rate'])
        
        best_loss = float('inf')
        patience = self.model_params['patience']
        patience_counter = 0
        
        print("Training PyTorch MLP...")
        for epoch in range(self.model_params['epochs']):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            avg_train_loss = train_loss / len(train_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.model_params['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best state? For now just keep current model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # CPUへ戻す（保存のため）
        self.model.cpu()

    def _set_feature_importance(self):
        if self.model_type == 'lgbm':
            importances = self.model.feature_importance(importance_type='gain')
        elif self.model_type in ['rf', 'gbc', 'xgb']:
            importances = self.model.feature_importances_
        elif self.model_type == 'catboost':
            importances = self.model.get_feature_importance()
        else:
            importances = np.zeros(len(self.feature_names))
            
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測確率を出力"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        # 特徴量の順序を合わせる
        X = X[self.feature_names]
        
        if self.model_type == 'lgbm':
            return self.model.predict(X)
        elif self.model_type in ['rf', 'gbc', 'xgb']:
            # [:, 1] で正例（勝利）の確率を取得
            return self.model.predict_proba(X)[:, 1]
        elif self.model_type == 'catboost':
            return self.model.predict_proba(X)[:, 1]
        elif self.model_type == 'pytorch_mlp':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                return self.model(X_tensor).numpy().flatten()
        
        return np.zeros(len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測確率を出力（predictと同じ）
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            予測確率の配列
        """
        return self.predict(X)
    
    def explain_prediction(self, X: pd.DataFrame, num_samples: int = 100) -> dict:
        """
        SHAP値を用いて予測の根拠を説明
        
        Args:
            X: 説明対象の特徴量DataFrame
            num_samples: SHAP計算に使用するサンプル数（背景データ）
            
        Returns:
            {
                'shap_values': SHAP値の配列 (n_samples × n_features),
                'base_value': ベース値（全体平均予測値）,
                'feature_names': 特徴量名リスト,
                'explanations': 各サンプルの説明（プラス/マイナス要因）
            }
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        if shap is None:
            raise ImportError("SHAP library is not installed. pip install shap")
        
        # 特徴量の順序を合わせる
        X = X[self.feature_names].copy()
        
        # SHAP Explainerの作成（モデルタイプに応じて選択）
        try:
            if self.model_type == 'lgbm':
                # TreeExplainerを使用（高速）
                explainer = shap.TreeExplainer(self.model)
                # check_additivity=Falseで数値誤差によるエラーを回避
                shap_values = explainer.shap_values(X, check_additivity=False)
                
                # 2値分類の場合、正例（1）のSHAP値を使用
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                base_value = explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[1]

            elif self.model_type in ['rf', 'gbc', 'xgb']:
                # TreeExplainerを使用
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X, check_additivity=False)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                    
                base_value = explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[1]
            else:
                # その他のモデルはKernelExplainer（やや低速）
                background = shap.sample(X, min(num_samples, len(X)))
                explainer = shap.KernelExplainer(self.predict, background)
                shap_values = explainer.shap_values(X)
                base_value = explainer.expected_value
                
            # base_valueのスカラ化（配列やリストの場合に対処）
            if isinstance(base_value, (list, np.ndarray)):
                if len(base_value) > 0:
                    base_value = base_value[0]
                else:
                    base_value = 0.5
            
            # 各サンプルのトップ要因を抽出
            explanations = []
            for i in range(len(X)):
                shap_row = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                
                # 寄与度の絶対値でソート
                indices = np.argsort(np.abs(shap_row))[::-1]
                
                positive_factors = []
                negative_factors = []
                
                for idx in indices:
                    feature_name = self.feature_names[idx]
                    shap_value = float(shap_row[idx])
                    feature_value = float(X.iloc[i, idx])
                    
                    factor_info = {
                        'feature': feature_name,
                        'value': feature_value,
                        'contribution': shap_value
                    }
                    
                    if shap_value > 0:
                        positive_factors.append(factor_info)
                    elif shap_value < 0:
                        negative_factors.append(factor_info)
                    
                    # トップ3まで
                    if len(positive_factors) >= 3 and len(negative_factors) >= 3:
                        break
                
                explanations.append({
                    'positive': positive_factors[:3],
                    'negative': negative_factors[:3]
                })
            
            return {
                'shap_values': shap_values,
                'base_value': float(base_value),
                'feature_names': self.feature_names,
                'explanations': explanations
            }
            
        except Exception as e:
            print(f"SHAP Error: {e}")
            # エラー時は空の結果を返す（UI側でハンドリング）
            return {
                'shap_values': [],
                'base_value': 0.5,
                'feature_names': self.feature_names,
                'explanations': []
            }
    
    def save(self, filepath: str = None):
        """
        モデルを保存
        
        Args:
            filepath: 保存先のパス
        """
        if filepath is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_DIR, 'horse_race_model.pkl')
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデル({self.model_type})を保存しました: {filepath}")
    
    def load(self, filepath: str = None):
        """
        モデルを読み込み
        
        Args:
            filepath: 読み込み元のパス
        """
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, 'horse_race_model.pkl')
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'lgbm')
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_params = model_data['model_params']
        
        print(f"モデルを読み込みました: {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        特徴量重要度を取得
        
        Args:
            top_n: 上位N個の特徴量
            
        Returns:
            特徴量重要度のDataFrame
        """
        if self.feature_importance is None:
            raise ValueError("モデルが学習されていません")
        
        return self.feature_importance.head(top_n)

    def debug_info(self):
        """デバッグ用：モデル内部情報を出力"""
        info = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'pandas_categorical': None
        }
        
        # LGBM固有情報
        if self.model_type == 'lgbm' and self.model is not None:
            try:
                # lgb.train() returns Booster directly, LGBMClassifier has booster_
                booster = self.model
                if hasattr(self.model, 'booster_'):
                    booster = self.model.booster_
                    
                if hasattr(booster, 'pandas_categorical'):
                    info['pandas_categorical'] = booster.pandas_categorical
                    
                # Store feature names from booster if available (more accurate)
                if hasattr(booster, 'feature_name'):
                     info['booster_feature_names'] = booster.feature_name()
            except Exception as e:
                info['error'] = str(e)
                
        return info


class RacePredictor:
    """レース予測クラス"""
    
    def __init__(self, model: HorseRaceModel, processor=None, engineer=None):
        """
        初期化
        
        Args:
            model: 学習済みのHorseRaceModel
            processor: DataProcessor
            engineer: FeatureEngineer
        """
        self.model = model
        self.processor = processor
        self.engineer = engineer
    
    def predict_race(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """
        レースの予測を行う
        
        Args:
            race_data: レースデータのDataFrame
            
        Returns:
            予測結果付きのDataFrame
        """
        df = race_data.copy()
        
        # 前処理
        if self.processor:
            df = self.processor.process_results(df)
        
        # 特徴量作成
        if self.engineer:
            df = self.engineer.add_jockey_features(df)
        
        # 予測に必要な特徴量を抽出
        feature_cols = [c for c in self.model.feature_names if c in df.columns]
        
        if len(feature_cols) < len(self.model.feature_names):
            print(f"Warning: 一部の特徴量が欠損しています")
            # 欠損特徴量は0で埋める
            for col in self.model.feature_names:
                if col not in df.columns:
                    df[col] = 0
        
        X = df[self.model.feature_names].copy()
        X = X.fillna(X.median())
        
        # Scaling if available
        if self.processor and self.processor.scaler:
             X = self.processor.transform_scale(X)
        
        # 予測
        df['予測確率'] = self.model.predict(X)
        df['予測順位'] = df['予測確率'].rank(ascending=False).astype(int)
        
        # 予測確率でソート
        df = df.sort_values('予測確率', ascending=False)
        
        return df
    
    def get_recommended_horses(self, race_data: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """
        推奨馬を取得
        
        Args:
            race_data: レースデータ
            top_n: 上位N頭
            
        Returns:
            推奨馬のDataFrame
        """
        predicted = self.predict_race(race_data)
        return predicted.head(top_n)


class EnsembleModel:
    """複数モデルのアンサンブルクラス"""
    
    def __init__(self, models: list = None):
        """
        初期化
        
        Args:
            models: HorseRaceModelのリスト
        """
        self.models = models or []
        self.weights = None
        self.feature_names = None
        if self.models:
            self.feature_names = self.models[0].feature_names
            self.weights = [1.0 / len(self.models)] * len(self.models)
            
    def add_model(self, model: HorseRaceModel, weight: float = 1.0):
        """モデルを追加"""
        self.models.append(model)
        if self.feature_names is None:
            self.feature_names = model.feature_names
        # 重みを再計算（均等）
        self.weights = [1.0 / len(self.models)] * len(self.models)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """アンサンブル予測（加重平均）"""
        if not self.models:
            raise ValueError("モデルが登録されていません")
            
        final_probs = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            final_probs += model.predict(X) * weight
            
        return final_probs
        
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        アンサンブルの特徴量重要度（平均）
        """
        if not self.models:
            raise ValueError("モデルが登録されていません")
            
        all_importances = []
        for model in self.models:
            try:
                imp = model.get_feature_importance(top_n=100) # 全特徴量取得
                # 正規化
                if imp['importance'].sum() > 0:
                    imp['importance'] = imp['importance'] / imp['importance'].sum()
                all_importances.append(imp)
            except:
                pass
                
        if not all_importances:
            return pd.DataFrame(columns=['feature', 'importance'])
            
        # 平均を計算
        combined = pd.concat(all_importances).groupby('feature')['importance'].mean().reset_index()
        return combined.sort_values('importance', ascending=False).head(top_n)

    def explain_prediction(self, X: pd.DataFrame, num_samples: int = 100) -> dict:
        """
        アンサンブルの予測根拠（各モデルのSHAP値の平均）
        """
        if not self.models:
             raise ValueError("モデルが登録されていません")
             
        # 代表モデル（最初の一つ）を使って枠組みを作成
        # 注: 本当は全モデルのSHAPを平均すべきだが、コストが高いので
        # 簡易的に最初のLGBM/RFモデルの結果を返す、もしくは加重平均する実装にする
        
        all_shap_values = []
        base_values = []
        feature_names = self.feature_names
        
        valid_explanations = 0
        
        for model in self.models:
            try:
                exp = model.explain_prediction(X, num_samples)
                all_shap_values.append(exp['shap_values'])
                base_values.append(exp['base_value'])
                valid_explanations += 1
            except Exception as e:
                print(f"Ensemble SHAP skip: {e}")
                
        if valid_explanations == 0:
            return {
                'shap_values': [],
                'base_value': 0.5,
                'feature_names': feature_names,
                'explanations': []
            }
            
        # 平均計算 (簡易実装: 次元が合う場合のみ)
        try:
            avg_shap = np.mean(all_shap_values, axis=0)
            avg_base = np.mean(base_values)
            
            # 説明文の再生成
            explanations = []
            for i in range(len(X)):
                shap_row = avg_shap[i] if len(avg_shap.shape) > 1 else avg_shap
                indices = np.argsort(np.abs(shap_row))[::-1]
                
                positive_factors = []
                negative_factors = []
                
                for idx in indices:
                    feature_name = feature_names[idx]
                    shap_value = float(shap_row[idx])
                    feature_value = float(X.iloc[i, idx])
                    
                    factor_info = {'feature': feature_name, 'value': feature_value, 'contribution': shap_value}
                    
                    if shap_value > 0: positive_factors.append(factor_info)
                    elif shap_value < 0: negative_factors.append(factor_info)
                    
                    if len(positive_factors) >= 3 and len(negative_factors) >= 3: break
                
                explanations.append({'positive': positive_factors[:3], 'negative': negative_factors[:3]})
                
            return {
                'shap_values': avg_shap,
                'base_value': avg_base,
                'feature_names': feature_names,
                'explanations': explanations
            }
        except:
            # 形状が合わない場合は最初のモデルの結果を返す
            return self.models[0].explain_prediction(X, num_samples)

    def save(self, dir_path: str = MODEL_DIR):
        """アンサンブルに含まれる全てのモデルを保存"""
        os.makedirs(dir_path, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(dir_path, f'model_{model.model_type}_{i}.pkl')
            model.save(path)
            
    def load(self, dir_path: str = MODEL_DIR):
        """ディレクトリ内の全モデルを読み込み"""
        self.models = []
        files = [f for f in os.listdir(dir_path) if f.startswith('model_') and f.endswith('.pkl')]
        for f in files:
            m = HorseRaceModel()
            m.load(os.path.join(dir_path, f))
            self.models.append(m)
        
        if self.models:
            self.feature_names = self.models[0].feature_names
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def explain_prediction(self, X: pd.DataFrame, num_samples: int = 100) -> dict:
        """
        アンサンブルモデルの予測を説明（各モデルの平均SHAP値）
        
        Args:
            X: 説明対象の特徴量DataFrame
            num_samples: SHAP計算に使用するサンプル数
            
        Returns:
            平均化されたSHAP値と説明情報
        """
        if not self.models:
            raise ValueError("モデルが登録されていません")
        
        # 各モデルのSHAP値を取得して平均化
        all_shap_values = []
        all_base_values = []
        
        for model in self.models:
            try:
                explanation = model.explain_prediction(X, num_samples)
                all_shap_values.append(explanation['shap_values'])
                all_base_values.append(explanation['base_value'])
            except Exception as e:
                print(f"Warning: SHAP calculation failed for {model.model_type}: {e}")
                continue
        
        if not all_shap_values:
            raise ValueError("全てのモデルでSHAP計算に失敗しました")
        
        # 平均SHAP値を計算
        avg_shap_values = np.mean(all_shap_values, axis=0)
        avg_base_value = np.mean(all_base_values)
        
        # 各サンプルのトップ要因を抽出（平均SHAP値を使用）
        explanations = []
        for i in range(len(X)):
            shap_row = avg_shap_values[i] if len(avg_shap_values.shape) > 1 else avg_shap_values
            
            indices = np.argsort(np.abs(shap_row))[::-1]
            
            positive_factors = []
            negative_factors = []
            
            for idx in indices:
                feature_name = self.feature_names[idx]
                shap_value = float(shap_row[idx])
                feature_value = float(X.iloc[i, idx])
                
                factor_info = {
                    'feature': feature_name,
                    'value': feature_value,
                    'contribution': shap_value
                }
                
                if shap_value > 0:
                    positive_factors.append(factor_info)
                elif shap_value < 0:
                    negative_factors.append(factor_info)
                
                if len(positive_factors) >= 3 and len(negative_factors) >= 3:
                    break
            
            explanations.append({
                'positive': positive_factors[:3],
                'negative': negative_factors[:3]
            })
        
        return {
            'shap_values': avg_shap_values,
            'base_value': float(avg_base_value),
            'feature_names': self.feature_names,
            'explanations': explanations
        }


def create_sample_model(model_type: str = 'lgbm'):
    """
    サンプルモデルを作成（デモ用）
    """
    # サンプルデータでモデルを作成
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        '枠番': np.random.randint(1, 9, n_samples),
        '馬番': np.random.randint(1, 19, n_samples),
        '斤量': np.random.uniform(50, 60, n_samples),
        '単勝': np.random.exponential(10, n_samples),
        '人気': np.random.randint(1, 19, n_samples),
        '年齢': np.random.randint(2, 8, n_samples),
        '体重': np.random.normal(480, 30, n_samples),
        '体重変化': np.random.normal(0, 10, n_samples),
        'course_len': np.random.choice([1200, 1400, 1600, 1800, 2000, 2400], n_samples),
        'avg_rank': np.random.uniform(3, 10, n_samples),
        'win_rate': np.random.uniform(0, 0.3, n_samples),
        'place_rate': np.random.uniform(0, 0.6, n_samples),
        'race_count': np.random.randint(1, 30, n_samples),
        'jockey_avg_rank': np.random.uniform(4, 10, n_samples),
        'jockey_win_rate': np.random.uniform(0, 0.2, n_samples),
        '性': np.random.randint(0, 3, n_samples),
        'race_type': np.random.randint(0, 3, n_samples),
        'venue_id': np.random.randint(1, 11, n_samples),
        'kai': np.random.randint(1, 6, n_samples),
        'day': np.random.randint(1, 13, n_samples),
        'race_num': np.random.randint(1, 13, n_samples),
        'avg_last_3f': np.random.uniform(33, 40, n_samples),
        'avg_running_style': np.random.uniform(0.1, 1.0, n_samples)
    })
    
    # 目的変数（人気と過去成績に基づいて生成）
    y = ((X['人気'] <= 3) & (X['avg_rank'] <= 5)).astype(int)
    y = y | (np.random.random(n_samples) < 0.1)  # ノイズ追加
    y = y.astype(int)
    
    model = HorseRaceModel(model_type=model_type)
    metrics = model.train(X, y)
    
    print("サンプルモデルの評価:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return model
