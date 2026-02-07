import sys
import pickle
import pandas as pd
import os
sys.path.append('.')
from modules.constants import RAW_DATA_DIR

path = os.path.join(RAW_DATA_DIR, 'return_tables.pickle')
with open(path, 'rb') as f:
    returns = pickle.load(f)

print("Sample Return Keys:", list(returns.keys())[:5])

sample_race = list(returns.keys())[0]
print(f"\nSample Race ({sample_race}) Data:")
df = returns[sample_race]
if isinstance(df, pd.Series):
    df = pd.DataFrame([df])
print(df)

print("\nUnique Bet Types in Returns:")
types = set()
for race_id in list(returns.keys())[:100]:
    r = returns[race_id]
    if isinstance(r, pd.Series):
        r = pd.DataFrame([r])
    for val in r[0].unique():
        types.add(val)
print(types)
