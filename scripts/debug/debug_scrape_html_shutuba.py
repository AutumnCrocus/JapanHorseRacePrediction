
import requests
from bs4 import BeautifulSoup
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from modules.constants import HEADERS, SHUTUBA_URL

def debug_shutuba_html(race_id):
    url = SHUTUBA_URL + str(race_id)
    print(f"Fetching {url}")
    response = requests.get(url, headers=HEADERS)
    response.encoding = "EUC-JP"
    
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"class": "Shutuba_Table"})
    
    if not table:
        print("Table not found")
        return
        
    rows = table.find_all("tr", {"class": "HorseList"})
    print(f"Found {len(rows)} horse rows.")
    
    if rows:
        row = rows[0]
        cols = row.find_all("td")
        print(f"Columns in first row: {len(cols)}")
        for i, col in enumerate(cols):
            class_ = col.get("class", [])
            text = col.text.strip().replace("\n", "")
            print(f"Col {i:02d}: Class={class_}, Text='{text[:30]}'")
            
            # spanなどを探索
            spans = col.find_all("span")
            for sp in spans:
                print(f"  Span Class={sp.get('class')}, Text='{sp.text.strip()}'")
                
            # 父・母探索
            if "父" in text or "母" in text:
                print(f"  *** Found Father/Mother keyword in Col {i} ***")

if __name__ == "__main__":
    debug_shutuba_html('202605010611')
