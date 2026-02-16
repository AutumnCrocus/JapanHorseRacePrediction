import pandas as pd
import pickle
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RETURN_FILE = os.path.join(DATA_DIR, 'return_tables.pickle')


def inspect():
    print("Loading return_tables.pickle...")
    with open(RETURN_FILE, 'rb') as f:
        data = pickle.load(f)

    print(f"Type: {type(data)}")
    print("Checking unique bet types...")

    if isinstance(data, pd.DataFrame):
        print(f"Index head: {data.index[:5]}")
        print(f"Index type: {data.index.dtype}")
        # Check first element with repr
        first_idx = data.index[0]
        rid = first_idx[0] if isinstance(first_idx, tuple) else first_idx
        print(f"First race_id repr: {repr(rid)}")

        # Check specific ID
        target_id = '202506010101'
        found = False
        for idx in data.index:
            rid = str(idx[0]) if isinstance(idx, tuple) else str(idx)
            if rid == target_id:
                found = True
                break
        print(f"Race {target_id} found in returns: {found}")
    elif isinstance(data, dict):
        bet_types = set()
        count = 0
        for race_id, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                bet_types.update(df[0].unique())
            count += 1
            if count > 1000: break # Check first 1000 races
            
        print(f"Unique Bet Types found: {bet_types}")


if __name__ == '__main__':
    inspect()
