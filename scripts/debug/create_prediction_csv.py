import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import argparse
import os
from modules.scraping import Shutuba, HorseResults
from modules.preprocessing import prepare_training_data

def main():
    parser = argparse.ArgumentParser(description='Create prediction CSV from Netkeiba Shutuba page')
    parser.add_argument('--race_id', type=str, required=True, help='Race ID (e.g., 202608010111)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    args = parser.parse_args()

    race_id = args.race_id
    if args.output:
        output_path = args.output
    else:
        output_path = f"prediction_{race_id}.csv"

    print(f"Fetching and processing data for Race ID: {race_id}...")
    try:
        from modules.data_loader import fetch_and_process_race_data
        final_df = fetch_and_process_race_data(race_id)
        
        print(f"Saving to {output_path}...")
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
