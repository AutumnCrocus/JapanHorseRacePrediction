
import sys
import os
import traceback

print("DEBUG WRAPPER: Starting...")
sys.stdout.flush()

try:
    # Add current dir to path so we can import scripts
    sys.path.append(os.getcwd())
    print(f"DEBUG WRAPPER: CWD is {os.getcwd()}")
    
    print("DEBUG WRAPPER: Importing simulate_2025_hybrid...")
    sys.stdout.flush()
    import scripts.simulate_2025_hybrid as sim
    
    print("DEBUG WRAPPER: Calling run_simulation()...")
    sys.stdout.flush()
    sim.run_simulation()
    
    print("DEBUG WRAPPER: Simulation finished successfully.")
except Exception as e:
    print(f"DEBUG WRAPPER: Exception occurred: {e}")
    traceback.print_exc()
finally:
    print("DEBUG WRAPPER: Exiting.")
    sys.stdout.flush()
