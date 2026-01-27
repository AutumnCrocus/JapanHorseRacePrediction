---
description: Verify frontend changes using automated checks
---

When you modify any frontend code (HTML/CSS/JS), you MUST verify the changes using the `validate_frontend.py` script BEFORE notifying the user. This ensures that no JavaScript errors are blocking the UI.

1. Ensure the Playwright environment is ready (install if needed).
2. Start the application server strictly for testing (use a different port if needed, e.g., 5001).
3. Run the validation script targeting the test server.

Example:
```bash
# 1. Start server in background (adjust path as needed)
python scripts/debug/run_app_test.py &

# 2. Wait for server (manual wait or check logic)

# 3. Run validation
python scripts/utils/validate_frontend.py http://127.0.0.1:5001/ "#featureChart"
```

If the script returns "FAILED", check the output for JS errors, fix them, and RE-RUN the test until it passes.
DO NOT claim the task is complete until this validaiton passes.
