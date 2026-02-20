import sys
import os
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.deepfm_inference import DeepFMInference

@patch('modules.deepfm_inference.os.path.exists', return_value=True)
@patch('modules.deepfm_inference.pickle.load')
@patch('modules.deepfm_inference.torch.load')
@patch('modules.deepfm_inference.DeepFM')
def test_deepfm_inference_predict(MockDeepFM, mock_torch_load, mock_pickle_load, mock_exists):
    """DeepFMInferenceがデータフレームを受け取り、正しく前処理と推論を行うかのテスト"""
    
    # LabelEncoderモックの準備
    mock_lbe_horse = MagicMock()
    mock_lbe_horse.classes_ = ['1', '2', 'unknown']
    mock_lbe_horse.transform.return_value = [1]
    
    mock_lbe_jockey = MagicMock()
    mock_lbe_jockey.classes_ = ['1001', 'unknown']
    mock_lbe_jockey.transform.return_value = [2]

    # メタデータのモック
    mock_pickle_load.return_value = {
        'feature_config': {
            'sparse': [{'name': 'horse_number'}, {'name': 'jockey_id'}],
            'dense': [{'name': 'weight'}, {'name': 'burden_weight'}]
        },
        'label_encoders': {
            'horse_number': mock_lbe_horse,
            'jockey_id': mock_lbe_jockey
        }
    }
    
    # モデルインスタンスのモック準備
    mock_model_instance = MagicMock()
    mock_model_instance.eval = MagicMock()
    
    # 推論結果のモック
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.85, 0.42])
    mock_model_instance.return_value = mock_tensor
    
    MockDeepFM.return_value = mock_model_instance
    
    # モックされたopen関数を使用して初期化
    with patch('builtins.open', MagicMock()):
        inference = DeepFMInference('dummy.pth', 'dummy.pkl', device='cpu')

    # 1. 空のデータフレームの場合
    df_empty = pd.DataFrame()
    preds_empty = inference.predict(df_empty)
    assert len(preds_empty) == 0
    
    # 2. 正常なデータフレームの場合
    df = pd.DataFrame({
        '馬番': [1, 2],
        '騎手': ['1001', '1002'],
        '体重': [480, 500],
        '斤量': [55, 56]
    })
    
    preds = inference.predict(df)
    
    # 出力の型と形状をチェック
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    assert preds[0] == 0.85
    assert preds[1] == 0.42
    
    # 内部のモデルが呼ばれたかチェック
    assert mock_model_instance.called
