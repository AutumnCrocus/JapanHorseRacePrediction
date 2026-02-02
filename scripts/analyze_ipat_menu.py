
import sys
import os
import time
from selenium.webdriver.common.by import By

# モジュールパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import load_ipat_credentials
from modules.ipat_direct_automator import IpatDirectAutomator

def analyze_menu_structure():
    print("=== IPAT Menu Structure Analysis ===")
    
    # 1. Credentials
    inetid, subscriber_no, pin, pars_no = load_ipat_credentials()
    
    automator = IpatDirectAutomator(debug_mode=True)
    
    try:
        # Login
        print("Logging in...")
        success, msg = automator.login(inetid, subscriber_no, pin, pars_no)
        if not success:
            print(f"Login Failed: {msg}")
            return

        print("Login Success. Waiting for menu URL...")
        
        # Explicitly wait for URL to contain target script
        try:
             WebDriverWait(automator.driver, 15).until(lambda d: "pw_890_i.cgi" in d.current_url)
             print("URL Verified: pw_890_i.cgi reached.")
        except:
             print("Timeout waiting for pw_890_i.cgi. Current URL:", automator.driver.current_url)
        
        time.sleep(5) # Wait for render
        
        # Save Screenshot (Top)
        automator._save_debug_screenshot(automator.driver, "menu_page_top")
        
        # Frame Analysis
        frames = automator.driver.find_elements(By.TAG_NAME, "frame")
        iframes = automator.driver.find_elements(By.TAG_NAME, "iframe")
        all_frames = frames + iframes
        
        print(f"Found {len(frames)} frames and {len(iframes)} iframes.")
        
        if len(all_frames) == 0:
            print("No frames found. Searching in main content...")
            # Check Images
            imgs = automator.driver.find_elements(By.TAG_NAME, "img")
            for img in imgs:
                alt = img.get_attribute("alt") or ""
                title = img.get_attribute("title") or ""
                if "通常" in alt or "投票" in alt or "通常" in title:
                    print(f"  [MATCH Main] Img: Alt='{alt}', Title='{title}'")
            
            # Check Anchors
            links = automator.driver.find_elements(By.TAG_NAME, "a")
            for link in links:
                text = link.text
                onclick = link.get_attribute("onclick") or ""
                title = link.get_attribute("title") or ""
                if "通常" in text or "投票" in text or "通常" in title or "bet" in onclick.lower():
                    print(f"  [MATCH Main] Link: Text='{text}', Title='{title}', OnClick='{onclick}'")

        for i, frame in enumerate(all_frames):
            print(f"--- Frame {i} ---")
            try:
                # Switch to frame
                automator.driver.switch_to.frame(i)
                print(f"Switched to Frame {i}")
                
                # Capture Source
                with open(f"debug_menu_frame_{i}.html", "w", encoding="utf-8") as f:
                    f.write(automator.driver.page_source)
                print(f"Saved debug_menu_frame_{i}.html")
                
                automator._save_debug_screenshot(automator.driver, f"menu_frame_{i}")
                
                # Search for Buttons
                print("  Searching for 'Normal Vote' candidates in Frame...")
                
                # Check Images
                imgs = automator.driver.find_elements(By.TAG_NAME, "img")
                for img in imgs:
                    alt = img.get_attribute("alt") or ""
                    title = img.get_attribute("title") or ""
                    src = img.get_attribute("src") or ""
                    if "通常" in alt or "投票" in alt or "通常" in title:
                        print(f"  [MATCH] Img: Alt='{alt}', Title='{title}', Src='{src}'")
                
                # Check Anchors
                links = automator.driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    text = link.text
                    onclick = link.get_attribute("onclick") or ""
                    title = link.get_attribute("title") or ""
                    if "通常" in text or "投票" in text or "通常" in title or "bet" in onclick.lower():
                        print(f"  [MATCH] Link: Text='{text}', Title='{title}', OnClick='{onclick}'")

                # Return to default content
                automator.driver.switch_to.default_content()
            except Exception as e:
                print(f"  Error accessing frame {i}: {e}")
                automator.driver.switch_to.default_content()

        print("\nAnalysis Complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        automator.close()

if __name__ == "__main__":
    analyze_menu_structure()
