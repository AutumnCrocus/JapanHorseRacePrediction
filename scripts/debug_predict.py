
import os
import sys
import pandas as pd
import numpy as np
import traceback

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.scraping import Shutuba, Odds
from scripts.predict_today_all import load_prediction_pipeline, process_race, load_historical_data

def debug_race(race_id):
    with open('debug_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"DEBUG: Starting process for {race_id}\n")
        
        try:
            predictor = load_prediction_pipeline()
            horse_results, peds = load_historical_data()
            
            f.write("Models loaded.\n")
            
            res = process_race(race_id, predictor, budget=5000, horse_results_db=horse_results, peds_db=peds)
            
            if res:
                f.write("DEBUG: process_race returned result.\n")
                f.write(f"DEBUG: Recommendations: {len(res['recommendations'])}\n")
                f.write(f"DEBUG: Predictions: {len(res['predictions'])}\n")
                f.write(str(res['predictions']) + "\n")
            else:
                f.write("DEBUG: process_race returned None.\n")
        
        except Exception as e:
            f.write(f"DEBUG: Exception in process_race: {e}\n")
            traceback.print_exc(file=f)

if __name__ == "__main__":
    target_id = "202610010412" 
    debug_race(target_id)
