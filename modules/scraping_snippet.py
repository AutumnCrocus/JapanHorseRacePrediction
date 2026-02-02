import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Define constants if missing
SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id="
HEADERS = {"User-Agent": "Mozilla/5.0"}

class Shutuba:
    """出馬表のスクレイピングクラス"""
    
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
            # &rf=race_submenu は必須ではないが、一応つけても良い。今回はシンプルに。
            response = requests.get(url, headers=HEADERS)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                print(f"Failed to fetch {url}: status code {response.status_code}")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", {"class": "Shutuba_Table"})
            if table is None:
                print("Shutuba table not found.")
                return pd.DataFrame()

            # レース情報取得
            race_info_text = ""
            race_list_item02 = soup.find("div", {"class": "RaceList_Item02"})
            if race_list_item02:
                race_info_text = race_list_item02.text.strip()
            
            # コース、距離、天候、馬場状態の解析
            # 例: "芝1600m (右 外 A) / 天候:曇 / 馬場:良"
            race_type = None
            course_len = None
            weather = None
            ground_state = None
            
            # コース・距離
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

            # 馬ごとのデータ取得
            horse_rows = table.find_all("tr", {"class": "HorseList"})
            data_list = []
            
            for row in horse_rows:
                # 1. 枠番
                waku_td = row.find("td", class_=re.compile("Waku"))
                waku = waku_td.text.strip() if waku_td else None
                
                # 2. 馬番
                umaban_td = row.find("td", class_=re.compile("Umaban"))
                umaban = umaban_td.text.strip() if umaban_td else None
                
                # 3. 馬名とID
                horse_td = row.find("td", {"class": "HorseInfo"})
                horse_name = None
                horse_id = None
                if horse_td:
                    h_link = horse_td.find("a")
                    if h_link:
                        horse_name = h_link.text.strip()
                        if "href" in h_link.attrs:
                            h_id_match = re.search(r"/horse/(\d+)", h_link["href"])
                            if h_id_match:
                                horse_id = h_id_match.group(1)

                # 4. 性齢
                sex_age_td = row.find("td", {"class": "Barei"})
                sex_age = sex_age_td.text.strip() if sex_age_td else None
                
                # 5. 斤量
                weight_td = row.find_all("td")[5] # 6番目のカラム（インデックス5）と推測
                # クラス指定がない場合が多いので位置で取得。ブラウザ解析ではnth-child(6)だった。
                # ただしBeautifulSoupのfind_all("td")の結果には非表示のものなどが含まれる可能性もあるため注意。
                # ブラウザエージェント結果: nth-child(6) -> つまり6番目。
                weight = weight_td.text.strip() if weight_td else None
                
                # 6. 騎手とID
                jockey_td = row.find("td", {"class": "Jockey"})
                jockey_name = None
                jockey_id = None
                if jockey_td:
                    j_link = jockey_td.find("a")
                    if j_link:
                        jockey_name = j_link.text.strip()
                        if "href" in j_link.attrs:
                            j_id_match = re.search(r"/jockey/.*(\d{5})", j_link["href"]) # ID形式要確認、URLは通常 .../result.html?id=01234 とか /jockey/01234/
                            # netkeiba PC: /jockey/05339/
                            # netkeiba SP: 
                            if not j_id_match:
                                j_id_match = re.search(r"/jockey/(\d+)", j_link["href"])
                            
                            if j_id_match:
                                jockey_id = j_id_match.group(1)

                # 7. 馬体重 (例: 502(0))
                horse_weight_td = row.find("td", {"class": "Weight"})
                horse_weight_raw = horse_weight_td.text.strip() if horse_weight_td else None
                
                # 8. オッズと人気
                odds_td = row.find("td", {"class": "Popular"}) # クラス名が変わる可能性あり、Txt_Rを含むか確認
                odds = None
                popularity = None
                
                # オッズ
                odds_span = row.find("span", {"class": "Odds_Ninki"}) # 修正: 出馬表では span class="Odds_Ninki" ? ブラウザエージェントの結果: td.Popular.Txt_R .Odds_Ninki
                if odds_span:
                    odds = odds_span.text.strip()
                
                # 人気
                pop_span = row.find("span", {"class": "Popular_Ninki"}) # .Popular_Ninki span
                # ブラウザエージェント結果では td.Popular_Ninki span
                pop_td = row.find("td", {"class": "Popular_Ninki"})
                if pop_td:
                    p_span = pop_td.find("span")
                    if p_span:
                        popularity = p_span.text.strip()

                data = {
                    "枠番": waku,
                    "馬番": umaban,
                    "馬名": horse_name,
                    "horse_id": horse_id,
                    "性齢": sex_age,
                    "斤量": weight,
                    "騎手": jockey_name,
                    "jockey_id": jockey_id,
                    "馬体重": horse_weight_raw,
                    "単勝": odds,
                    "人気": popularity,
                    "race_type": race_type,
                    "course_len": course_len,
                    "weather": weather,
                    "ground_state": ground_state
                }
                data_list.append(data)
                
            df = pd.DataFrame(data_list)
            return df
            
        except Exception as e:
            print(f"Error scraping shutuba {race_id}: {e}")
            return pd.DataFrame()
