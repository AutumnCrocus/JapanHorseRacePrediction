
import numpy as np
import choix.opt

# Dummy data
n_items = 5
rankings = [[0, 1, 2], [3, 4]]
params = np.zeros(n_items)

try:
    # Check signature and return
    # Usually: opt_rankings(params, data, alpha) ?
    # or opt_rankings(rankings, n_items) returns a function?
    
    # Based on dir struct: choix.opt has opt_rankings.
    # Let's try to call it with params and data.
    res = choix.opt.opt_rankings(params, rankings, alpha=0.0)
    print(f"Result type: {type(res)}")
    if isinstance(res, tuple):
        print(f"Tuple len: {len(res)}")
        print(f"Val: {res[0]}")
        print(f"Grad shape: {res[1].shape}")
except Exception as e:
    print(f"Error calling opt_rankings: {e}")
    # Try getting docstring
    print(choix.opt.opt_rankings.__doc__)
