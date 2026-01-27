
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../..'))

from app import app, load_model

if __name__ == '__main__':
    print("Starting test server on port 5001...")
    load_model()
    # Disable debug reloader to prevent creating child processes difficult to kill
    app.run(debug=False, host='127.0.0.1', port=5001)
