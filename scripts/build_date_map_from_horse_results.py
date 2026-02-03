
import pandas as pd
import pickle
import os
import sys
import re

# Add project root to path
sys.path.append(os.getcwd())

from modules.constants import RAW_DATA_DIR, HORSE_RESULTS_FILE, DATA_DIR

# Manual Place Map just in case constants is missing coverage or issues
PLACE_MAP = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10'
}

def build_map():
    hr_path = os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE)
    output_path = os.path.join(DATA_DIR, "date_map_2025.pickle")
    
    print(f"Loading {hr_path}...")
    with open(hr_path, 'rb') as f:
        df = pickle.load(f)
    
    # Normalize columns
    df.columns = df.columns.str.replace(' ', '')
    print("Columns:", df.columns.tolist())
    
    # Check required columns
    if '日付' not in df.columns or '開催' not in df.columns:
        print("Error: Missing '日付' or '開催' column.")
        return

    # Filter for 2025
    # Date format usually YYYY/MM/DD
    df = df.dropna(subset=['日付', '開催'])
    df = df[df['日付'].astype(str).str.contains('2025')]
    
    print(f"Found {len(df)} records for 2025.")
    
    date_map = {} # Key: YYYYSSKKDD, Value: YYYY-MM-DD
    
    # Iterate unique (Date, Kaisai) pairs
    unique_pairs = df[['日付', '開催']].drop_duplicates()
    
    count = 0
    for _, row in unique_pairs.iterrows():
        date_str = str(row['日付']) # 2025/01/05
        kaisai = str(row['開催']) # 1回中山1日
        
        # Parse date -> YYYY-MM-DD
        date_std = date_str.replace('/', '-')
        
        # Parse Kaisai -> Kai, Place, Day
        # Regex: (\d+)回(..)(\d+)日
        match = re.search(r'(\d+)回(.+?)(\d+)日', kaisai)
        if match:
            kai = int(match.group(1))
            place_name = match.group(2)
            day = int(match.group(3))
            
            place_code = PLACE_MAP.get(place_name)
            if not place_code:
                # Fuzzy match
                for p, c in PLACE_MAP.items():
                    if p in place_name:
                        place_code = c
                        break
            
            if place_code:
                # Key: YYYY SS KK DD
                key = f"2025{place_code}{kai:02d}{day:02d}"
                date_map[key] = date_std
                count += 1
            else:
                pass # Local race or unknown place
        else:
            pass # Not a standard JRA Kaisai string (e.g. Local)
            
    print(f"Mapped {len(date_map)} unique days.")
    
    with open(output_path, 'wb') as f:
        pickle.dump(date_map, f)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    build_map()
