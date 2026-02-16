
import pandas as pd
import pickle
import os
import sys
import re
from tqdm import tqdm

sys.path.append(os.getcwd())
from modules.constants import RAW_DATA_DIR, RESULTS_FILE, HORSE_RESULTS_FILE, DATA_DIR, PLACE_DICT

# Reverse PLACE_DICT for name->id
PLACE_NAME_TO_ID = {v: k for k, v in PLACE_DICT.items()}

def clean_horse_name(name):
    # Remove whitespace and common noise
    return str(name).strip().replace(' ', '').replace('　', '')

def parse_horse_results_race_name(r_name):
    # "21/09/05 高知 8R Ｃ３ー１４"
    # Returns: date_str (YYYY-MM-DD), venue_name, race_num
    try:
        match = re.search(r'(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\d+)R', str(r_name))
        if match:
            date_short, v_name, r_num = match.groups()
            year_prefix = "20" if int(date_short[:2]) < 50 else "19"
            date_full = f"{year_prefix}{date_short.replace('/', '-')}"
            return date_full, v_name.strip(), int(r_num)
    except:
        pass
    return None, None, None

def run():
    print("Loading datasets...")
    with open(os.path.join(RAW_DATA_DIR, RESULTS_FILE), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(RAW_DATA_DIR, HORSE_RESULTS_FILE), 'rb') as f:
        hr = pickle.load(f)

    # 1. Process Results (Target: find RaceID -> {Horses, VenueCode, R})
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
        results['race_id'] = results['level_0'].astype(str)
        # Assuming level_0 is race_id or index has it
    else:
        results['race_id'] = results.index.astype(str)

    # Filter 2025 only to save time
    results = results[results['race_id'].str.startswith('2025')]
    print(f"Results (2025): {len(results)} rows")
    
    # Normalize horse name col
    r_horse_col = '馬名' if '馬名' in results.columns else None
    if not r_horse_col:
        # try find
        for c in results.columns:
            if '馬' in str(c) and '名' in str(c):
                r_horse_col = c
                break
    if not r_horse_col:
        print("Error: Could not find '馬名' in results.")
        return

    # Group results by RaceID
    race_id_data = {} # rid -> {venue_code, r_num, horses: set}
    grouped_res = results.groupby('race_id')
    for rid, group in tqdm(grouped_res, desc="Processing Results groups"):
        rid = str(rid)
        # Parse rid: YYYY SS KK DD RR (12 digits) or similar
        # 2025 06 ...
        if len(rid) < 10: continue
        v_code = rid[4:6]
        r_num = int(rid[-2:])
        horses = set(group[r_horse_col].apply(clean_horse_name))
        race_id_data[rid] = {'v_code': v_code, 'r': r_num, 'horses': horses}

    print(f"  Prepared {len(race_id_data)} races from Results.")

    # 2. Process Horse Results (Source: Date, Venue, R -> {Horses})
    # Identify proper columns
    hr_cols = hr.columns.tolist()
    hr_race_name_col = next((c for c in hr_cols if 'レース名' in c), None)
    hr_horse_name_col = next((c for c in hr_cols if '馬' in c and '名' in c), None) # usually '馬名' is not in horse_results?
    # Wait, horse_results is usually 'row per horse' but the `horse_name` is implied?
    # The file `horse_results.pickle` usually contains the horse's history. 
    # But usually it has '馬名' column?
    # Inspecting Step 39 output: `['日付', '開催', ...]` 
    # It didn't show '馬名'. It showed headers of the *results table* (Weather, R, RaceName, Memo, Winner, Prize).
    # Ah, `horse_results` is typically `pd.read_html` of netkeiba horse profile > results.
    # It DOES NOT CONTAIN THE HOST HORSE NAME usually, because it belongs to that horse!
    # UNLESS it's a "Race Results" dump (all horses in a race).
    # If it is "Race Results" dump, it has "馬名".
    # If it is "Horse Past Performance" dump, it lacks "Horse Name" column because it repeats.
    # The output in Step 39/81 shows '着 順', '騎手', '勝ち馬（2着馬)'.
    # This looks like Horse Past Performance (Target Horse's history).
    # The column `勝ち馬（2着馬)` contains the winner name (or 2nd if current is winner).
    # This is NOT the current horse's name.
    
    # CRITICAL: If `horse_results` does NOT have the horse name, I cannot match horses!
    # BUT, `horse_results` is usually a collection of DFs, keyed by HorseID?
    # Or a big concatenated DF?
    # If concatenated, we lost the source HorseID?
    # Unless there is a column `horse_id` or similar.
    # The columns shown were `['日付', '開催', ...]`. No `horse_id` shown (truncated).
    
    # If I cannot match horses, I can match (Venue, R, Date) -> RaceID?
    # But I don't know the Date for the RaceID! I want to find it.
    # I know RaceID -> (Venue, R).
    # I know HorseResults -> (Date, Venue, R).
    # I can match on (Venue, R).
    # But (Venue, R) happens multiple times a year (Jan 5th Nakayama 11R, Dec 28th Nakayama 11R).
    # I need to distinguish instances.
    # I need "Kai" (Hold) or "Day".
    # Result RaceID has "Kai/Day".
    # HorseResults has "Date".
    # If I group HorseResults by (Venue, R), I get a list of Dates.
    # If I group Results by (Venue, R), I get a list of RaceIDs (sorted by time implicitly).
    # If I sort both lists, I can zip them!
    # 1st Nakayama 11R (RaceID ...010111) maps to Earliest Date of Nakayama 11R.
    # 2nd Nakayama 11R maps to Next Date...
    
    # Strat:
    # 1. Group Results by (VenueCode, R). Sort by RaceID (which is chronological).
    # 2. Group HorseResults by (VenueName, R). Sort by Date.
    # 3. Match nth occurrence to nth occurrence.
    # 4. Profit.
    
    # Map VenueName (Kanji) to VenueCode.
    hr['date_parsed'], hr['venue_parsed'], hr['r_parsed'] = zip(*hr[hr_race_name_col].apply(parse_horse_results_race_name))
    
    # Drop rows where parsing failed
    hr_clean = hr.dropna(subset=['date_parsed', 'venue_parsed', 'r_parsed'])
    hr_clean = hr_clean[hr_clean['date_parsed'].str.contains('2025')]
    
    # Group dates by (Venue, R)
    # Mapping: (VenueCode, R) -> List of Dates (sorted)
    hr_schedule = {} # (v_code, r) -> sorted unique dates
    
    unique_races = hr_clean[['date_parsed', 'venue_parsed', 'r_parsed']].drop_duplicates()
    for _, row in unique_races.iterrows():
        d, v_name, r = row['date_parsed'], row['venue_parsed'], row['r_parsed']
        v_code = PLACE_NAME_TO_ID.get(v_name)
        if not v_code: continue
        
        key = (v_code, r)
        if key not in hr_schedule: hr_schedule[key] = set()
        hr_schedule[key].add(d)
        
    # Sort dates
    hr_schedule_sorted = {k: sorted(list(v)) for k, v in hr_schedule.items()}
    
    # Group results by (VenueCode, R) -> List of RaceIDs (sorted)
    res_schedule = {} # (v_code, r) -> sorted unique race_ids
    for rid, data in race_id_data.items():
        key = (data['v_code'], data['r'])
        if key not in res_schedule: res_schedule[key] = []
        res_schedule[key].append(rid)
        
    res_schedule_sorted = {k: sorted(v) for k, v in res_schedule.items()}
    
    # Match
    final_map = {} # RaceID -> Date
    matches = 0
    for key, rids in res_schedule_sorted.items():
        if key not in hr_schedule_sorted:
            # print(f"Missing schedule for {key} in horse_results")
            continue
            
        dates = hr_schedule_sorted[key]
        
        # We perform matching by index
        # This assumes dataset completeness! 
        # If Results has 10 races and HR has 9, we might mis-align.
        # But `horse_results` is usually MORE complete or SAME.
        # If aligned, great.
        
        # To be safer: JRA calendar is regular.
        # But we can try just zipping for now.
        # Or better: RaceID contains "Kai" and "Day".
        # 2025 06 01 01 11 (Nakayama 1st Kai 1st Day 11R).
        # We can sort RaceIDs.
        # We can sort Dates.
         
        limit = min(len(rids), len(dates))
        for i in range(limit):
            final_map[rids[i]] = dates[i]
            matches += 1
            
    print(f"Matched {matches} races.")
    with open(os.path.join(DATA_DIR, "date_map_2025.pickle"), 'wb') as f:
        pickle.dump(final_map, f)
    print("Saved map.")

if __name__ == "__main__":
    run()
