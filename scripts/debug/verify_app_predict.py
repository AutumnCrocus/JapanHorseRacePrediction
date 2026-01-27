"""
Webアプリの予測ロジック検証スクリプト
app.py の run_prediction_logic を直接呼び出し、エラーを再現・修正確認する。
"""
import sys
import os
import pandas as pd
import numpy as np
from flask import Flask

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# app.py から関数をインポートするために、app.py の構造に依存しないようにモックするか、
# app.py をインポートしてテストする。
# app.py はトップレベルで実行されることを想定している部分があるかもしれないが、
# 関数単位でインポートできればベスト。
from app import run_prediction_logic, app


from app import run_prediction_logic, app, get_model

def test_prediction_logic():
    print("Loading model...")
    # Need to load model within app context or manually
    with app.app_context():
        # Force model load
        model = get_model()
        if model is None:
            print("Failed to load model.")
            return

        debug_info = model.debug_info()
        feature_names = debug_info.get('feature_names', [])
        
        print(f"Model Features: {len(feature_names)}")
        print(feature_names)
        
        # カテゴリ定義の確認
        cats = debug_info.get('pandas_categorical', [])
        if cats:
            print(f"Categorical definitions found: {len(cats)} sets")
            for i, c in enumerate(cats):
                print(f"  Cat {i} length: {len(c)}")
                # 必要なら中身も表示
                if len(c) < 20: print(f"    {c}")
        else:
            print("No categorical definitions found via debug_info.")

    # 5頭分 (学習データ不一致を狙う) のデータを作成
    # Feature names must match exactly what model expects
    data = []
    for i in range(5):
        row = {col: 0 for col in feature_names}
        # 適当な値を設定
        row['枠番'] = (i // 2) + 1
        row['馬番'] = i + 1
        row['単勝'] = 10.0 + i
        row['人気'] = i + 1
        row['馬名'] = f"TestHorse{i+1}"
        
        # カテゴリカル変数の値がカテゴリ定義外にならないように注意、あるいはわざと外す
        # ここでは一般的な値を入れる
        
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # 属性設定
    df.attrs['race_name'] = "Test Race"
    df.attrs['race_data01'] = "Test Data 01"
    df.attrs['race_data02'] = "Test Data 02"
    
    print("Running prediction logic...")
    try:
        with app.app_context():
            # Run prediction
            # This logic in app.py currently tries to set categories manually.
            # We want to see if that manual setting works or fails against the model.
            response = run_prediction_logic(df, "Test Race", "Info", race_id="2026TEST", budget=2000)
            
            if hasattr(response, 'get_json'):
                json_data = response.get_json()
            else:
                json_data = response.json
                
                
            print("Success!")
            print(f"Predictions: {len(json_data['predictions'])}")
            print(f"Recommendations: {len(json_data['recommendations'])}")
            print(f"Confidence Level: {json_data.get('confidence_level', 'Not Found')}")
            
    except Exception as e:
        print("\n!!! ERROR DETECTED !!!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_logic()
