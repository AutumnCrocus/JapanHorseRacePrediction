
import sys
import os
import json
from flask import Flask, render_template

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../..'))

from app import app, get_model

def verify_injection():
    print("Verifying server-side data injection...")
    
    with app.test_request_context('/'):
        # Ensure model is loaded
        get_model()
        
        # Helper to simulate the index route logic
        # We can't call app.view_functions['index']() directly easily if it renders template,
        # but we can simulate the logic or request the client.
        
        client = app.test_client()
        response = client.get('/')
        
        html = response.data.decode('utf-8')
        
        print(f"Status Code: {response.status_code}")
        
        if 'window.INITIAL_MODEL_DATA' in html:
            print("SUCCESS: 'window.INITIAL_MODEL_DATA' found in HTML.")
            
            # Extract the JSON part to verify it's valid
            start_marker = 'window.INITIAL_MODEL_DATA = '
            end_marker = ';\n'
            
            try:
                start_idx = html.find(start_marker) + len(start_marker)
                end_idx = html.find(';', start_idx)
                json_str = html[start_idx:end_idx]
                
                data = json.loads(json_str)
                print("SUCCESS: Injected data is valid JSON.")
                print(f"Algorithm: {data.get('algorithm')}")
                print(f"Features: {len(data.get('features'))} items")
                print(f"Metrics: {data.get('metrics')}")
                
            except Exception as e:
                print(f"ERROR: Failed to parse injected JSON: {e}")
                print(f"Snippet: {html[start_idx:start_idx+100]}...")
        else:
            print("FAILURE: 'window.INITIAL_MODEL_DATA' NOT found in HTML.")
            print("Snippet of HTML head:")
            print(html[:1000])

if __name__ == "__main__":
    verify_injection()
