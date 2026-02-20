import os
import glob

RAW_DIR = r'c:\Users\t4kic\Documents\JapanHorseRacePrediction\data\raw'

years = [2024, 2025]
categories = ['results', 'payouts', 'shutuba']

print("=== Raw Data Check ===")
for year in years:
    print(f"\nYear: {year}")
    for cat in categories:
        path = os.path.join(RAW_DIR, cat, str(year))
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.pickle")) + glob.glob(os.path.join(path, "*.pkl")) + glob.glob(os.path.join(path, "*.html"))
            print(f"  {cat}: {len(files)} files found in {path}")
        else:
            print(f"  {cat}: DIRECTORY MISSING ({path})")
