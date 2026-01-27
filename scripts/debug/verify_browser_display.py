
import asyncio
from playwright.async_api import async_playwright
import json
import os
import sys

# Screenshot directory
ARTIFACT_DIR = r"C:\Users\t4kic\.gemini\antigravity\brain\3ee8c28d-ec14-4236-b68a-45aff9c77235"

async def check_url(page, url, label):
    print(f"\n--- Checking {label} ({url}) ---")
    
    # Capture console logs
    page.on("console", lambda msg: print(f"BROWSER_LOG: {msg.text}"))
    page.on("pageerror", lambda err: print(f"BROWSER_ERROR: {err}"))
    
    try:
        await page.goto(url, wait_until="networkidle")
        
        # 1. Check title
        title = await page.title()
        print(f"Title: {title}")
        
        # 2. Check injected data details
        injected_data = await page.evaluate("() => window.INITIAL_MODEL_DATA")
        if injected_data:
            print("SUCCESS: window.INITIAL_MODEL_DATA found.")
            print(f"Data dump: {json.dumps(injected_data, ensure_ascii=False)[:200]}...") # Show beginning of data
        else:
            print("FAILURE: window.INITIAL_MODEL_DATA NOT found.")
            
        # 3. Check Chart Element and Scroll
        chart_el = await page.query_selector('#featureChart')
        if chart_el:
            print("SUCCESS: #featureChart element found. Scrolling to view...")
            await chart_el.scroll_into_view_if_needed()
            # Wait for animation or rendering
            await page.wait_for_timeout(1000)
            
            # Alternative: Scroll to specific query if chart is hidden or needs offset
            # await page.evaluate("document.getElementById('analytics').scrollIntoView()")
        else:
            print("FAILURE: #featureChart element NOT found.")
            
        # 4. Screenshot
        screenshot_path = os.path.join(ARTIFACT_DIR, f"screenshot_{label}.png")
        await page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
        
    except Exception as e:
        print(f"Error checking {url}: {e}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Check User's Port (5000)
        await check_url(page, "http://127.0.0.1:5000/", "user_port_5000")
        
        # Check Test Port (5001)
        await check_url(page, "http://127.0.0.1:5001/", "test_port_5001")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
