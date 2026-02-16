
print("Verifying imports...", flush=True)
import os
# Optional: test with and without thread restriction to see if it's truly fixed
# os.environ['OMP_NUM_THREADS'] = '1' 

try:
    import lightgbm
    print(f"LightGBM imported: {lightgbm.__version__}", flush=True)
except Exception as e:
    print(f"LightGBM Failed: {e}", flush=True)

try:
    import sklearn
    print(f"Sklearn imported: {sklearn.__version__}", flush=True)
except Exception as e:
    print(f"Sklearn Failed: {e}", flush=True)
