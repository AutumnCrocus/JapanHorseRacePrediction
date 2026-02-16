import sys
import os
print("Hello from check_env")
sys.path.append(os.getcwd())
try:
    from modules.ipat_direct_automator import IpatDirectAutomator
    print("Import Successful")
except Exception as e:
    print(f"Import Failed: {e}")
