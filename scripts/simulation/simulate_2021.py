"""
2021年シミュレーションスクリプト (期待値戦略)
・学習済みモデル(pure_model.pkl)を使用
・2021年の全レースを対象に予測
・EV = 予測勝率(score) * 確定単勝オッズ
・EV > 閾値 の馬を購入し、回収率を計算
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import MODEL_DIR, RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, PEDS_FILE
from modules.preprocessing import prepare_training_data
from modules.training import HorseRaceModel
from scripts.train_period import train_period_model # インポート確認用

def simulate_2021():
    print("=== 2021年 投資シミュレーション (EV戦略) ===")
    
    # 1. モデル読み込み
    model_path = os.path.join(MODEL_DIR, 'pure_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print("Loading model...")
    model = HorseRaceModel()
    model.load(model_path)
    
    # processor/engineer もロードしたいが、
    # 学習時に保存されたものは train_model_improved.py で上書きされている可能性がある。
    # train_period.py では保存していない（model.saveのみ）。
    # しかし、前処理ロジックは preprocessing.py にあるので、
    # 再度 prepare_training_data を呼べば同じ特徴量が作れるはず。
    
    # 2. データ読み込み
    print("Loading data...")
    results_path = os.path.join(RAW_DATA_DIR, RESULTS_FILE)
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    peds_path = os.path.join(RAW_DATA_DIR, PEDS_FILE)
    
    results_df = pd.read_pickle(results_path)
    hr_df = pd.read_pickle(hr_path) if os.path.exists(hr_path) else None
    peds_df = pd.read_pickle(peds_path) if os.path.exists(peds_path) else None
    
    # 2021年のみ抽出
    # prepare_training_data の後に絞るのが安全（ラグ特徴量のため）
    
    # 3. 前処理
    print("Preprocessing...")
    # scale=False で統一
    ret = prepare_training_data(results_df, hr_df, peds_df, scale=False)
    
    df_full = None
    if len(ret) >= 7:
        X, y, processor, engineer, bias_map, jockey_stats, df_full = ret[:7]
    elif len(ret) >= 6:
        X, y, processor, engineer, bias_map, jockey_stats = ret[:6]
    else:
        # Fallback
        vals = ret
        X, y = vals[0], vals[1]
    
    # オッズ情報を保持しておく (シミュレーション用)
    # Xには odds カラムがあるはず（prepare_training_dataで追加されるから）
    
    simulation_df = X.copy()
    simulation_df['target'] = y
    
    # Indexから年を取得
    if df_full is not None and 'date' in df_full.columns:
        simulation_df['year'] = df_full['date'].dt.year.values
        # 着順情報も結合しておく (シミュレーション用)
        if '着順' in df_full.columns:
            simulation_df['rank_res'] = df_full['着順'].values
        elif 'rank_num' in df_full.columns:
             simulation_df['rank_res'] = df_full['rank_num'].values
    elif 'date' in results_df.columns:
         try:
            simulation_df['year'] = results_df.loc[X.index, 'date'].dt.year
         except:
            simulation_df['year'] = simulation_df.index.astype(str).str[:4].astype(int)
    else:
         simulation_df['year'] = simulation_df.index.astype(str).str[:4].astype(int)

    # 2021年フィルタ
    target_year = 2021
    sim_data = simulation_df[simulation_df['year'] == target_year].copy()
    
    print(f"Target Races (2021): {len(sim_data)} rows")
    if sim_data.empty:
        print("No data for 2021.")
        return

    # 4. 予測
    use_cols = [c for c in model.feature_names if c in sim_data.columns]
    
    print(f"Predicting using {len(use_cols)} features...")
    
    X_test = sim_data[use_cols].copy()
            
    # カテゴリカル変換
    for col in ['枠番', '馬番']:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
            
    probs = model.predict(X_test)
    sim_data['score'] = probs
    
    # 5. シミュレーション
    print("\nSimulating betting...")
    
    # パラメータ
    EV_THRESHOLD = 1.0 
    BET_AMOUNT = 1000
    
    total_invest = 0
    total_return = 0
    bet_count = 0
    hit_count = 0
    
    if 'odds' not in sim_data.columns:
        print("Error: 'odds' column missing.")
        return
        
    sim_data = sim_data.dropna(subset=['odds'])
    sim_data['EV'] = sim_data['score'] * sim_data['odds']
    
    # 購入対象
    bet_target = sim_data[sim_data['EV'] >= EV_THRESHOLD]
    
    for idx, row in bet_target.iterrows():
        total_invest += BET_AMOUNT
        bet_count += 1
        
        hit = False
        # rank_res があればそれを使う
        if 'rank_res' in row:
            try:
                rank = float(row['rank_res'])
                if rank == 1:
                    hit = True
            except:
                pass
        
        if hit:
            hit_count += 1
            ret = BET_AMOUNT * row['odds']
            total_return += ret
            
    print(f"\nResults (2021) - EV Threshold: {EV_THRESHOLD}")
    print(f"Bet Count: {bet_count} / {len(sim_data)} horses")
    print(f"Invest: {total_invest:,} Yen")
    print(f"Return: {total_return:,} Yen")
    
    rate = (total_return / total_invest * 100) if total_invest > 0 else 0
    print(f"Recovery Rate: {rate:.1f}%")
    print(f"Hit Rate: {(hit_count/bet_count*100) if bet_count>0 else 0:.1f}%")

if __name__ == '__main__':
    simulate_2021()
