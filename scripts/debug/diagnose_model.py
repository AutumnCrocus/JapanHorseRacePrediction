"""
モデル診断スクリプト
LightGBMモデルが保持しているカテゴリカル変数の定義（pandas_categorical）を出力する。
"""
import sys
import os
import pickle
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.training import HorseRaceModel

def diagnose():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'production_model.pkl')
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    print(f"Loaded object type: {type(model_data)}")
    if isinstance(model_data, dict):
        model = model_data['model'] # LGBMClassifier or Booster
        print(f"Extracted model type: {type(model)}")
    else:
        # 古い形式か、直接保存された場合
        model = model_data
    
    print(f"Model object type: {type(model)}")
    
    booster = None
    if hasattr(model, 'booster_'):
        booster = model.booster_
    elif hasattr(model, 'pandas_categorical'):
        # model itself is booster-like
        booster = model
    
    if booster is None:
        print("Could not retrieve booster or categorical info.")
        # Try finding it in dict? no.
        return

    print("\n=== Model Categorical Info ===")
    cats_list = getattr(booster, 'pandas_categorical', None)
    
    if cats_list is None:
        print("No pandas_categorical info found in booster.")
    else:
        print("pandas_categorical found.")
        for i, cats in enumerate(cats_list):
            print(f"Categorical Feature Index {i}:")
            print(f"  Length: {len(cats)}")
            print(f"  Dtype: {type(cats)}")
            print(f"  Content: {cats}") # Print content to see if 0 is there

    # Feature Names
    print("\n=== Feature Names ===")
    if hasattr(model, 'feature_name_'):
        print(model.feature_name_)
    elif hasattr(model, 'feature_name'):
        print(model.feature_name())

if __name__ == "__main__":
    diagnose()
