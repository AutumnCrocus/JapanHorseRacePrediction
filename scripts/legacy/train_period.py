"""
モデル学習スクリプト (期間指定 & オッズ除外版)
・学習期間: 2016-2020年
・特徴量: オッズ、人気、回収率（結果依存）を除外
・目的: 純粋な能力予測モデルの構築
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# モジュールパス追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel

def train_period_model():
    print("=== モデル学習 (期間指定: 2016-2020) ===")
    
    # 1. データ読み込み
    print("[1/5] データ読み込み...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} が見つかりません。")
        return

    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 期間フィルタリング (2016-2020)
    # results_df の index (race_id) から年を抽出
    try:
        years = results_df.index.astype(str).str[:4].astype(int)
        # 学習には2016-2020を使う。
        # ただし、前処理(Lag features)のために少し前のデータも必要かもしれないが、
        # prepare_training_data内部で時系列処理されるため、
        # ここで絞ってしまうとLagが計算できない可能性がある。
        # -> 全データを前処理に通した後でフィルタリングするのが正解。
        
        # 検証(2021)のデータリークを防ぐため、2021以降のデータは学習データには一切入れない。
        # 2021以降のデータもロードして前処理すると、Expanding Windowなどで情報が漏れる可能性がある？
        # -> Expanding Windowは shift(1) しているので直前のレースまでしか見ない。
        # -> 2021年のデータを前処理に含めても、2020年以前のデータ作成には影響しない（未来を見ないなら）。
        # -> しかし、FeatureEngineerの stats 計算などが全データでの平均を含んでしまうとリークになる。
        # -> preprocessing.pyの実装は expanding().mean().shift(1) なのでリーク回避されているはず。
        # -> 念のため、学習に使うのは「2020年までのデータで構築された特徴量」であればよい。
        
        pass 
    except Exception as e:
        print(f"Error parsing years: {e}")
        return

    print(f"Total Results: {len(results_df)} rows")
    
    # 2. 前処理 & 特徴量生成
    print("\n[2/5] 前処理 & 特徴量生成...")
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    if len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
    elif len(ret) == 5:
        X, y, processor, engineer, bias_map = ret
        jockey_stats = None
    else:
        vals = ret
        X, y, processor, engineer = vals[0], vals[1], vals[2], vals[3]
        bias_map = None
        jockey_stats = None

    # カテゴリカル変換
    for col in ['枠番', '馬番']:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # オッズ関連特徴量の除外 (重要)
    # odds, popularity, jockey_return_avg を削除
    exclude_cols = ['odds', 'popularity', 'jockey_return_avg']
    drop_cols = [c for c in exclude_cols if c in X.columns]
    
    if drop_cols:
        print(f"Dropping odds-related features: {drop_cols}")
        X = X.drop(columns=drop_cols)
        
        # feature_namesも更新が必要だが、ModelクラスがX.columnsを見るならOK
        # 保存されるengineer/processorには「使わないカラム」として記録されないといけないので
        # 後で predict する時に不整合が起きないよう注意が必要。
        # -> predict_tomorrow.py では 'odds' が生成されるが、モデル入力時に X[model.feature_names] でフィルタされる。
        # -> つまり、学習時の model.feature_names に odds が入っていなければ、予測時に odds があっても無視される。OK。

    # Indexから年を再取得（XのIndexはresults_dfと同じはず）
    try:
        data_years = X.index.astype(str).str[:4].astype(int)
    except:
        # dateカラムから取得
        if 'date' in results_df.columns:
             data_years = results_df.loc[X.index, 'date'].dt.year
        else:
             print("Cannot determine years.")
             return

    # 3. データ分割 (Period Split)
    print("\n[3/5] データ分割 (2016-2020 for Training)...")
    
    train_mask = (data_years >= 2016) & (data_years <= 2020)
    # 検証用は学習データの中の直近（例えば2020年）を使うか、
    # 学習期間内の時系列スプリットで評価するか。
    # ユーザー指示は「2016-2020学習、2021検証」。
    # 2021検証は別スクリプトでやるので、ここでは学習時のEarly Stopping用のValが必要。
    # 2020年をValにする。
    
    real_train_mask = (data_years >= 2016) & (data_years <= 2019)
    val_mask = (data_years == 2020)
    
    X_train = X[real_train_mask]
    y_train = y[real_train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"Train (2016-2019): {len(X_train)}")
    print(f"Val   (2020)     : {len(X_val)}")
    
    # 4. 学習
    print("\n[4/5] モデル学習 (Pure Ability Model)...")
    model = HorseRaceModel()
    
    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    print("\n--- Validation Metrics (2020) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Feature Importance
    if hasattr(model.model, 'feature_importance'):
        imp = model.model.feature_importance(importance_type='gain')
        names = model.model.feature_name()
        imp_df = pd.DataFrame({'feature': names, 'gain': imp}).sort_values('gain', ascending=False)
        print("\nTop 15 Feature Importance (No Odds):")
        print(imp_df.head(15))

    # 5. 保存 (モデル名を変える)
    print("\n[5/5] モデル保存...")
    # pure_model として保存
    model_name = 'pure_model.pkl'
    
    # processor/engineer は共通で良いが、もし上書きすると
    # 以前のモデル(oddsあり)が動かなくなる？
    # -> FeatureEngineerなどはロジックを持つだけで状態（学習済みパラメータ）はあまりない。
    # -> ただし、Target Encoding的なもの（Jockey Statsなど）は engineer.pkl に保持されているかも。
    # -> preprocessing.pyのExpanding windowは都度計算なのでpklには入らない。
    # -> jockey_stats は返り値として別保存している。
    # -> つまり engineer.pkl は安全。
    
    model.save(os.path.join(MODEL_DIR, model_name))
    
    # 今回の学習で使った特徴量リストなどを記録しておくべきだが、
    # Modelオブジェクト内に feature_names が保存されるので大丈夫。
    
    print(f"Saved to {model_name}")

if __name__ == '__main__':
    train_period_model()
