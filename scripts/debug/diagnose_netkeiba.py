
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def run_diagnosis():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    
    print("Initializing Driver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        # 1. Inspect Odds Page (Quinella)
        url = "https://race.netkeiba.com/odds/index.html?type=b4&race_id=202610010111&housiki=c0"
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for table
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='checkbox']")))
            print("Odds table loaded.")
        except:
            print("Timeout waiting for odds table.")
            return

        # Inspect All Checkbox IDs to find pattern
        print("--- Inspecting ALL Checkbox IDs ---")
        try:
             checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
             print(f"Total checkboxes found: {len(checkboxes)}")
             for i, cb in enumerate(checkboxes[:30]): # Print first 30
                 print(f"CB {i}: ID='{cb.get_attribute('id')}', Name='{cb.get_attribute('name')}', Value='{cb.get_attribute('value')}'")
        except Exception as e:
             print(f"Error listing checkboxes: {e}")

        time.sleep(1)

        time.sleep(1)
            
        # Check for AddBtn
        add_btns = driver.find_elements(By.CSS_SELECTOR, ".AddBtn, button.AddBtn")
        if add_btns:
            print(f"Found {len(add_btns)} .AddBtn elements.")
            for btn in add_btns:
                print(f"AddBtn: Text='{btn.text}', Visible={btn.is_displayed()}")
                if btn.is_displayed():
                     driver.execute_script("arguments[0].click();", btn)
                     print("Clicked AddBtn via JS.")
        else:
            print("No .AddBtn found.")



        # 2. Inspect IPAT Page (The one provided by user)
        ipat_url = "https://race.netkeiba.com/ipat/ipat.html?date=20260124&race_id=202610010111&return_url=dispatch.html%3Fmode%3Dbet_rewrite%26race_id%3D202610010111"
        print(f"Navigating to IPAT URL: {ipat_url}")
        driver.get(ipat_url)
        time.sleep(2)
        
        # Dump inputs
        inputs = driver.find_elements(By.TAG_NAME, "input")
        print(f"Found {len(inputs)} inputs on IPAT page.")
        for i in inputs[:20]: # Print first 20
            print(f"Input: ID={i.get_attribute('id')}, Name={i.get_attribute('name')}, Type={i.get_attribute('type')}")
            
        # Look for amount fields specifically
        # They often look like 'guide_amount_1' or similar in netkeiba?
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    run_diagnosis()
