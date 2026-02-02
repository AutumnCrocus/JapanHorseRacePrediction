import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

# Define constants if missing, though likely imported elsewhere or defined here
SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id="
HEADERS = {"User-Agent": "Mozilla/5.0"}

class ShutubaMethod:
    @staticmethod
    def scrape(race_id: str) -> pd.DataFrame:
        """
        出馬表をスクレイピング
        
        Args:
            race_id: レースID
            
        Returns:
            出馬表データのDataFrame
        """
        try:
            url = SHUTUBA_URL + str(race_id)
            response = requests.get(url, headers=HEADERS)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                print(f"Failed to fetch {url}: status code {response.status_code}")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", {"class": "Shutuba_Table"})
            if table is None:
                # スマホ版対応などのフォールバックが必要な場合はここに追記
                # 今回はPC版/レスポンシブHTML前提
                print("Shutuba table not found.")
                # class名が変わっている可能性も考慮して少し緩く探す手もあるが一旦返す
                return pd.DataFrame()

            # レース情報取得 (ヘッダー部分)
            # 例: 3歳未勝利, 芝2000m...
            # Selectors based on Browser inspection
            race_info_text = ""
            
            # .RaceList_Item02 contains "芝2200m (右 外) 天候:曇 馬場:重"
            r_item02 = soup.find("div", {"class": "RaceList_Item02"})
            if r_item02:
                race_info_text += r_item02.text.strip()
            
            # コース情報の解析
            race_type = None
            course_len = None
            weather = None
            ground_state = None
            
            # 芝/ダ/障 + 距離
            course_match = re.search(r"(芝|ダ|障)(\d+)m", race_info_text)
            if course_match:
                race_type = course_match.group(1)
                course_len = int(course_match.group(2))
            
            # 天候
            if "晴" in race_info_text: weather = "晴"
            elif "曇" in race_info_text: weather = "曇"
            elif "小雨" in race_info_text: weather = "小雨"
            elif "雨" in race_info_text: weather = "雨"
            elif "小雪" in race_info_text: weather = "小雪"
            elif "雪" in race_info_text: weather = "雪"
            
            # 馬場
            if "不良" in race_info_text: ground_state = "不良"
            elif "稍重" in race_info_text: ground_state = "稍重"
            elif "重" in race_info_text: ground_state = "重"
            elif "良" in race_info_text: ground_state = "良"

            # 馬リスト
            horse_rows = table.find_all("tr", {"class": "HorseList"})
            data_list = []
            
            for row in horse_rows:
                # 1. 枠番
                waku_td = row.find("td", class_=re.compile(r"Waku"))
                waku = waku_td.text.strip() if waku_td else None
                
                # 2. 馬番
                umaban_td = row.find("td", class_=re.compile(r"Umaban"))
                umaban = umaban_td.text.strip() if umaban_td else None
                
                # 3. 馬名
                horse_td = row.find("td", {"class": "HorseInfo"})
                horse_name = None
                horse_id = None
                if horse_td:
                    h_link = horse_td.find("a")
                    if h_link:
                        horse_name = h_link.text.strip()
                        if "href" in h_link.attrs:
                            # href="/horse/2020104866"
                            found = re.search(r"/horse/(\d+)", h_link["href"])
                            if found:
                                horse_id = found.group(1)
                
                # 4. 性齢
                sex_age_td = row.find("td", {"class": "Barei"})
                sex_age = sex_age_td.text.strip() if sex_age_td else None
                
                # 5. 斤量
                # nth-child(6) usually
                cols = row.find_all("td")
                weight = None
                if len(cols) > 5:
                    weight = cols[5].text.strip()
                
                # 6. 騎手
                jockey_td = row.find("td", {"class": "Jockey"})
                jockey_name = None
                jockey_id = None
                if jockey_td:
                    j_link = jockey_td.find("a")
                    if j_link:
                        jockey_name = j_link.text.strip()
                        if "href" in j_link.attrs:
                            found = re.search(r"/jockey/.*(\d{5})", j_link["href"])
                            if not found:
                                found = re.search(r"/jockey/(\d+)", j_link["href"])
                            if found:
                                jockey_id = found.group(1)
                
                # 7. 馬体重
                h_weight_td = row.find("td", {"class": "Weight"})
                h_weight = h_weight_td.text.strip() if h_weight_td else None
                
                # 8. オッズと人気 (予測オッズ)
                # td.Popular.Txt_R .Odds_Ninki
                odds = None
                pop = None
                
                odds_span = row.find("span", id=re.compile(r"odds-\d+")) # id="odds-1" etc sometimes?
                # Using class
                odds_elem = row.find("span", {"class": "Odds_Ninki"})
                if odds_elem:
                    odds = odds_elem.text.strip()
                
                pop_elem = row.find("span", {"class": "Popular_Ninki"})
                if pop_elem:
                    pop = pop_elem.text.strip()
                # Fallback to td if span not found
                if not pop and row.find("td", {"class": "Popular_Ninki"}):
                    pop = row.find("td", {"class": "Popular_Ninki"}).text.strip()
                
                data_list.append({
                    "枠番": waku,
                    "馬番": umaban,
                    "馬名": horse_name,
                    "horse_id": horse_id,
                    "性齢": sex_age,
                    "斤量": weight,
                    "騎手": jockey_name,
                    "jockey_id": jockey_id,
                    "馬体重": h_weight,
                    "単勝": odds,
                    "人気": pop,
                    "race_type": race_type,
                    "course_len": course_len,
                    "weather": weather,
                    "ground_state": ground_state
                })
            
            return pd.DataFrame(data_list)
            
        except Exception as e:
            print(f"Error scraping shutuba {race_id}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
