import pandas as pd
import os

def extract_strategy_b():
    # Check both root and data/
    paths = ['rolling_prediction_details.csv', 'data/rolling_prediction_details.csv']
    file_path = None
    for p in paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        print("File not found in root or data/ directory.")
        return

    print(f"Reading from: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='shift_jis')

    print("Columns:", df.columns.tolist())
    
    # Strategy B: Score >= 0.4
    # Assuming Single Win (Tansho) bet based on previous analysis logic
    strat_b = df[df['score'] >= 0.4].copy()
    
    print(f"\nTotal Bets: {len(strat_b)}")
    
    # Select relevant columns
    # Adjust column names based on actual CSV content
    cols = ['date', 'venue_id', 'race_num', 'horse_number', 'horse_name', 'score', 'odds', 'rank', 'is_win']
    available_cols = [c for c in cols if c in df.columns]
    
    result_df = strat_b[available_cols].copy()
    
    # Add bet type and amount
    result_df['bet_type'] = '単勝'
    result_df['amount'] = 100
    
    # Rename for clarity
    result_df = result_df.rename(columns={
        'horse_number': '馬番',
        'horse_name': '馬名',
        'score': 'AIスコア',
        'odds': 'オッズ',
        'rank': '着順'
    })
    
    # Show first 20 rows
    print("\n--- Strategy B Betting History (First 20) ---")
    print(result_df.head(20).to_string())
    
    # Save to CSV for user
    output_path = 'data/strategy_b_details.csv'
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved details to {output_path}")

if __name__ == "__main__":
    extract_strategy_b()
