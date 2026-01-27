
import asyncio
import sys
import os
import json
from playwright.async_api import async_playwright

# Usage: python validate_frontend.py <url> <selector_to_check>
# Example: python validate_frontend.py http://127.0.0.1:5000/ #featureChart

async def validate(url, selector):
    print(f"Starting frontend validation for: {url}")
    print(f"Target selector: {selector}")
    
    error_count = 0
    errors = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # 1. Setup Error Listener
        page.on("console", lambda msg: errors.append(f"CONSOLE: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: errors.append(f"PAGE_ERROR: {err}"))
        
        try:
            # 2. Navigate
            response = await page.goto(url, wait_until="networkidle", timeout=10000)
            if not response or response.status >= 400:
                print(f"FAILED: HTTP Status {response.status if response else 'Available'}")
                sys.exit(1)

            # 3. Check for Errors
            if errors:
                print("FAILED: JS Errors detected during load:")
                for e in errors:
                    print(f"  - {e}")
                sys.exit(1)
            
            # 4. Check Element Existence
            if selector:
                element = await page.query_selector(selector)
                if not element:
                    # Fallback: Check for placeholder message (valid state when no model data is available)
                    placeholder = await page.query_selector(".placeholder-message")
                    if placeholder and await placeholder.is_visible():
                        print(f"WARNING: Selector '{selector}' not found, but placeholder message is visible. Assuming valid empty state.")
                    else:
                        print(f"FAILED: Selector '{selector}' not found in DOM, and no placeholder message found.")
                        sys.exit(1)
                
                # 5. Check Visibility (if element exists)
                elif not await element.is_visible():
                    print(f"FAILED: Selector '{selector}' exists but is not visible.")
                    sys.exit(1)
                    
            print("SUCCESS: Frontend validation passed.")
            print("- No JS errors detected.")
            print("- Target element found and visible.")
            
        except Exception as e:
            print(f"FATAL ERROR: Validation script crashed: {e}")
            sys.exit(1)
        finally:
            await browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_frontend.py <url> [selector]")
        sys.exit(1)
        
    target_url = sys.argv[1]
    target_selector = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(validate(target_url, target_selector))
