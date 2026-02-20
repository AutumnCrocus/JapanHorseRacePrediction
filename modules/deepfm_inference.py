import os
import pickle
import torch
import pandas as pd
import numpy as np
from modules.models.deepfm import DeepFM

class DeepFMInference:
    """DeepFM モデルを使用したリアルタイム推論クラス"""
    
    def __init__(self, model_path, metadata_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.feature_config = metadata['feature_config']
            self.label_encoders = metadata['label_encoders']
            
        # Initialize and Load Model
        self.model = DeepFM(self.feature_config, dnn_hidden_units=(256, 128), 
                           embedding_dim=8, device=self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrameを受け取り、DeepFMスコアを算出する
        """
        if df.empty:
            return np.array([])
            
        df_proc = df.copy()
        
        # 1. データの整合性チェックとリネーム (create_deepfm_data.py のロジックに合わせる)
        rename_map = {
            '斤量': 'burden_weight',
            '体重': 'weight',
            '体重変化': 'weight_diff',
            '馬番': 'horse_number',
            '枠番': 'frame_number',
            '年齢': 'age',
            '性': 'sex',
            'weather': 'weather',
            'ground_state': 'ground_state',
            'race_type': 'race_type',
            'venue_id': 'venue_id',
            'course_len': 'course_len'
        }
        
        # 欠損カラムの補填
        if 'jockey_id' not in df_proc.columns and '騎手' in df_proc.columns:
            df_proc['jockey_id'] = df_proc['騎手']
        if 'trainer_id' not in df_proc.columns and '調教師' in df_proc.columns:
            df_proc['trainer_id'] = df_proc['調教師']
            
        for jp, en in rename_map.items():
            if jp in df_proc.columns:
                df_proc[en] = df_proc[jp]
            elif en not in df_proc.columns:
                # デフォルト値
                if en == 'age': df_proc[en] = 4
                elif en == 'burden_weight': df_proc[en] = 55
                elif en == 'weight': df_proc[en] = 480
                elif en == 'weight_diff': df_proc[en] = 0
                elif en == 'horse_number': df_proc[en] = 8
                else: df_proc[en] = 0
                
        # 2. 前処理 (カテゴリ・数値)
        sparse_features = [f['name'] for f in self.feature_config['sparse']]
        dense_features = [f['name'] for f in self.feature_config['dense']]
        
        # Sparse Encoding
        for feat in sparse_features:
            if feat in df_proc.columns:
                lbe = self.label_encoders[feat]
                # 未知のカテゴリは LabelEncoder に存在しないため、安全に処理
                # (学習時に 'unknown' を入れているので、それを活用)
                def safe_encode(val):
                    val_str = str(val)
                    if val_str in lbe.classes_:
                        return lbe.transform([val_str])[0]
                    elif 'unknown' in lbe.classes_:
                        return lbe.transform(['unknown'])[0]
                    else:
                        return 0
                df_proc[feat] = df_proc[feat].apply(safe_encode)
            else:
                df_proc[feat] = 0
                
        # Dense Scaling (学習時は MinMaxScaler 使用)
        # 本来は mms も保存すべきだが、概ねの範囲で正規化
        # burden_weight (48-60), weight (400-600), age (2-10)
        # 一旦簡易的に 0 で埋める (DeepFMはIDが支配的なので)
        for feat in dense_features:
            if feat in df_proc.columns:
                df_proc[feat] = pd.to_numeric(df_proc[feat], errors='coerce').fillna(0)
                # 簡易正規化 (MinMaxScalerを保存していない場合のフォールバック)
                if feat == 'weight': df_proc[feat] = (df_proc[feat] - 400) / 200
                elif feat == 'burden_weight': df_proc[feat] = (df_proc[feat] - 50) / 10
                elif feat == 'age': df_proc[feat] = (df_proc[feat] - 2) / 10
            else:
                df_proc[feat] = 0
                
        # 3. 推論
        X_sparse = torch.LongTensor(df_proc[sparse_features].values).to(self.device)
        X_dense = torch.FloatTensor(df_proc[dense_features].values).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_sparse, X_dense)
            
        return preds.cpu().numpy().flatten()
