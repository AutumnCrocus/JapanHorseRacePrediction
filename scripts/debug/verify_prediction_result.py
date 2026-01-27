import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from main import predict_race
import pandas as pd
import sys

try:
    df = predict_race(file_path="prediction_202606010111.csv")
    print("Columns:", df.columns.tolist())
    
    cols_to_show = ["馬番", "馬名", "単勝", "人気", "予測順位"]
    # Check if prob column exists
    if "pred_prob" in df.columns:
        cols_to_show.append("pred_prob")
    elif "score" in df.columns:
        cols_to_show.append("score")
        
    print(df.sort_values("予測順位")[cols_to_show].to_string(index=False))
    
    df.to_csv("prediction_result_202606010111.csv", index=False, encoding='utf-8-sig')
    print("Saved result to prediction_result_202606010111.csv")
except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
