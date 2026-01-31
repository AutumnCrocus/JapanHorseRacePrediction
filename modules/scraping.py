"""
スクレイピングモジュール
netkeiba.comからレース結果、馬の成績、血統データを収集
"""

import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from .constants import (
    RACE_URL, HORSE_URL, HORSE_PED_URL, SHUTUBA_URL, HEADERS, REQUEST_INTERVAL, MAX_WORKERS
)


class Results:
    """レース結果のスクレイピングクラス"""
    
    @staticmethod
    def _scrape_single_race(race_id, session):
        """単一のレース結果を取得（内部用）"""
        try:
            url = RACE_URL + str(race_id)
            response = session.get(url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                return None, True # (df, failed)

            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", {"class": "ResultsByRaceDetail"})
            if table is None:
                return None, True

            try:
                df = pd.read_html(StringIO(str(table)))[0]
            except ValueError:
                return None, False

            rows = table.find_all("tr")
            rows = [r for r in rows if r.find("td")]
            
            horse_id_list, jockey_id_list, trainer_id_list = [], [], []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 4: continue
                
                h_link = cols[3].find("a")
                horse_id = re.findall(r"\d+", h_link["href"])[0] if h_link and "href" in h_link.attrs else None
                horse_id_list.append(horse_id)
                    
                if len(cols) > 6:
                    j_link = cols[6].find("a")
                    jockey_id = re.findall(r"\d+", j_link["href"])[0] if j_link and "href" in j_link.attrs else None
                    jockey_id_list.append(jockey_id)
                else: jockey_id_list.append(None)

                if len(cols) > 14:
                     t_link = cols[14].find("a")
                     trainer_id = re.findall(r"\d+", t_link["href"])[0] if t_link and "href" in t_link.attrs else None
                     trainer_id_list.append(trainer_id)
                else: trainer_id_list.append(None)

            df = df.iloc[len(df)-len(rows):]
            df["horse_id"] = horse_id_list
            df["jockey_id"] = jockey_id_list
            df["trainer_id"] = trainer_id_list
            
            info_text = ""
            race_info_div = soup.find("div", {"class": "RaceData01"})
            race_data_div = soup.find("div", {"class": "RaceData02"})
            if race_info_div: info_text += race_info_div.text
            if race_data_div: info_text += " " + race_data_div.text
                
            if info_text:
                course_info = re.findall(r"(芝|ダ|障)(\d+)m", info_text)
                if course_info:
                    df["race_type"] = course_info[0][0]
                    df["course_len"] = int(course_info[0][1])
                
                for w in ["晴", "曇", "小雨", "雨", "雪"]:
                    if w in info_text:
                        df["weather"] = w
                        break
                for g in ["良", "稍重", "不良", "重"]:
                    if g in info_text:
                        df["ground_state"] = g
                        break
                
                title = soup.find("title").text if soup.find("title") else ""
                date_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", title)
                if date_match: df["date"] = date_match.group(1)
            
            df.index = [race_id] * len(df)
            return df, False
        except Exception as e:
            print(f"Error scraping race {race_id}: {e}")
            return None, False

    @staticmethod
    def _scrape_single_day(day_race_ids, session, pbar):
        """一日のレース結果を逐次取得（1R失敗でスキップ）"""
        results = {}
        for i, race_id in enumerate(day_race_ids):
            df, failed = Results._scrape_single_race(race_id, session)
            if failed:
                # 1R失敗時はその日の残りもスキップ
                is_race_01 = str(race_id).endswith("01")
                if is_race_01:
                    skip_count = len(day_race_ids) - (i + 1)
                    pbar.update(1 + skip_count)
                    break
            
            if df is not None:
                results[race_id] = df
            pbar.update(1)
            time.sleep(REQUEST_INTERVAL)
        return results

    @staticmethod
    def scrape(race_id_list: list, limit: int = None) -> pd.DataFrame:
        """レース結果を並列スクレイピング（会場・日単位）"""
        # 日単位（最初の10桁が同じもの）でグループ化
        days = []
        current_day = []
        last_day_id = ""
        for rid in race_id_list:
            day_id = str(rid)[:10]
            if day_id != last_day_id:
                if current_day: days.append(current_day)
                current_day = [rid]
                last_day_id = day_id
            else:
                current_day.append(rid)
        if current_day: days.append(current_day)

        results_dict = {}
        session = requests.Session()
        pbar = tqdm(total=len(race_id_list), desc="レース結果取得")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_day = {executor.submit(Results._scrape_single_day, day_ids, session, pbar): day_ids for day_ids in days}
            
            for future in as_completed(future_to_day):
                day_results = future.result()
                results_dict.update(day_results)
                if limit and len(results_dict) >= limit:
                    break
        
        pbar.close()
        if not results_dict: return pd.DataFrame()
        
        results_df = pd.concat(results_dict.values())
        return results_df


class HorseResults:
    """馬の過去成績のスクレイピングクラス"""
    
    @staticmethod
    def _scrape_single_horse(horse_id, session):
        """単一の馬の成績を取得（並列実行用ヘルパー）"""
        try:
            url = HORSE_URL + str(horse_id)
            response = session.get(url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                return horse_id, None

            dfs = pd.read_html(StringIO(response.text))
            
            # 過去成績テーブルを特定
            for df in dfs:
                if "着順" in df.columns or "着 順" in df.columns:
                    df.index = [horse_id] * len(df)
                    return horse_id, df
            return horse_id, None
        except Exception:
            return horse_id, None

    @staticmethod
    def scrape(horse_id_list: list) -> pd.DataFrame:
        """馬の過去成績をスクレイピング (並列実行対応)"""
        horse_results_dict = {}
        session = requests.Session()
        
        # tqdmで進捗を表示しつつ並列実行
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_horse = {executor.submit(HorseResults._scrape_single_horse, h_id, session): h_id for h_id in horse_id_list}
            
            for future in tqdm(as_completed(future_to_horse), total=len(horse_id_list), desc="馬成績取得"):
                horse_id, df = future.result()
                if df is not None:
                    horse_results_dict[horse_id] = df
        
        if not horse_results_dict:
            return pd.DataFrame()
        
        horse_results_df = pd.concat(horse_results_dict.values())
        return horse_results_df


class Peds:
    """血統データのスクレイピングクラス"""
    
    @staticmethod
    def _scrape_single_ped(horse_id, session):
        """単一の血統データを取得（並列実行用ヘルパー）"""
        try:
            url = HORSE_PED_URL + str(horse_id)
            response = session.get(url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                return horse_id, None

            # 血統テーブルは class="blood_table" を持つ
            dfs = pd.read_html(StringIO(response.text), attrs={"class": "blood_table"})
            if dfs:
                df = dfs[0]
                # 血統テーブルをフラット化 (5代血統: 2+4+8+16+32 = 62頭分)
                # 左から右へ、上から下へという順序でフラット化される
                peds_list = df.values.flatten().tolist()
                
                # 重複や不要なデータを除去しつつ、リスト形式で保持
                # NaNを空文字に置換
                peds_list = [str(p) if pd.notna(p) else "" for p in peds_list]
                return horse_id, peds_list
            return horse_id, None
        except Exception:
            return horse_id, None

    @staticmethod
    def scrape(horse_id_list: list) -> pd.DataFrame:
        """血統データをスクレイピング (並列実行対応)"""
        peds_dict = {}
        session = requests.Session()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ped = {executor.submit(Peds._scrape_single_ped, h_id, session): h_id for h_id in set(horse_id_list)}
            
            for future in tqdm(as_completed(future_to_ped), total=len(set(horse_id_list)), desc="血統データ取得"):
                horse_id, peds_list = future.result()
                if peds_list is not None:
                    peds_dict[horse_id] = peds_list
        
        if not peds_dict:
            return pd.DataFrame()
        
        # DataFrame化 (列名は peds_0, peds_1, ... とする)
        peds_df = pd.DataFrame.from_dict(peds_dict, orient="index")
        peds_df.columns = [f"peds_{i}" for i in range(len(peds_df.columns))]
        return peds_df




class Return:
    """払戻データのスクレイピングクラス"""
    
    @staticmethod
    def _scrape_single_return(race_id, session):
        """単一の払戻データを取得（並列実行用ヘルパー）"""
        try:
            url = RACE_URL + str(race_id)
            response = session.get(url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                return race_id, None

            soup = BeautifulSoup(response.text, "lxml")
            payback_div = soup.find("div", {"class": "PayBackHost"}) or soup.find("div", {"class": "RaceResult_Return"})
            
            if payback_div:
                dfs = pd.read_html(StringIO(str(payback_div)))
                if dfs:
                     merged_df = pd.concat(dfs)
                     return race_id, merged_df
            else:
                dfs = pd.read_html(StringIO(response.text))
                for df in dfs:
                     if "単勝" in str(df.values) or "複勝" in str(df.values):
                         return race_id, df
            return race_id, None
        except Exception:
            return race_id, None

    @staticmethod
    def scrape(race_id_list: list) -> pd.DataFrame:
        """払戻データをスクレイピング (並列実行対応)"""
        return_dict = {}
        session = requests.Session()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_return = {executor.submit(Return._scrape_single_return, r_id, session): r_id for r_id in race_id_list}
            
            for future in tqdm(as_completed(future_to_return), total=len(race_id_list), desc="払戻データ取得"):
                race_id, df = future.result()
                if df is not None:
                    return_dict[race_id] = df
        
        if not return_dict:
            return pd.DataFrame()
        
        return_df = pd.concat(return_dict.values(), keys=return_dict.keys())
        return return_df


def get_race_id_list(start_year: int, end_year: int = None, place_codes: list = None) -> list:
    """レースIDのリストを生成 (期間指定対応)"""
    if end_year is None:
        end_year = start_year
    
    if place_codes is None:
        place_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    
    race_id_list = []
    for year in range(start_year, end_year + 1):
        for place in place_codes:
            for kai in range(1, 7):
                for day in range(1, 13):
                    for race in range(1, 13):
                        race_id = f"{year}{place}{kai:02d}{day:02d}{race:02d}"
                        race_id_list.append(race_id)
    return race_id_list


def update_data(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """既存データに新しいデータを追加（重複を除去）"""
    if old_df.empty: return new_df
    if new_df.empty: return old_df
    filtered_old = old_df[~old_df.index.isin(new_df.index)]
    return pd.concat([filtered_old, new_df])


def get_race_date_info(race_id: str) -> dict:
    """
    出馬表ページから開催日情報を取得
    
    Args:
        race_id: レースID (例: 202601050111)
    
    Returns:
        dict: {
            'date': '2026年1月31日',
            'dow': '土',  # 曜日
            'dow_paren': '(土)'  # 括弧付き曜日
        }
        情報が取得できなかった場合は空の辞書を返す
    """
    try:
        url = SHUTUBA_URL + str(race_id)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.encoding = "EUC-JP"
        
        if response.status_code != 200:
            print(f"Failed to fetch race date info for {race_id}: status {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # 年月日をtitleタグから取得
        # 例: "白富士Ｓ(L) 出馬表 | 2026年1月31日 東京11R..."
        full_date = None
        title_elem = soup.find("title")
        if title_elem:
            title_text = title_elem.text
            date_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日)", title_text)
            if date_match:
                full_date = date_match.group(1)
        
        # 曜日をdd.Active aセレクタから取得
        # 例: "1月31日(土)"
        dow = None
        dow_paren = None
        active_date_elem = soup.select_one("dd.Active a")
        if active_date_elem:
            active_date_text = active_date_elem.text.strip()
            # 曜日を抽出 (括弧付き)
            dow_match = re.search(r"\((.)\)", active_date_text)
            if dow_match:
                dow = dow_match.group(1)
                dow_paren = f"({dow})"
        
        # フォールバック: titleから曜日を取得できなかった場合
        if not dow and full_date:
            # 日付文字列から曜日を計算
            try:
                # "2026年1月31日" -> datetime
                date_obj = pd.to_datetime(full_date, format="%Y年%m月%d日")
                dow_list = ["月", "火", "水", "木", "金", "土", "日"]
                dow = dow_list[date_obj.weekday()]
                dow_paren = f"({dow})"
            except:
                pass
        
        result = {}
        if full_date:
            result['date'] = full_date
        if dow:
            result['dow'] = dow
        if dow_paren:
            result['dow_paren'] = dow_paren
        
        return result
        
    except Exception as e:
        print(f"Error getting race date info for {race_id}: {e}")
        return {}


class Shutuba:
    """出馬表のスクレイピングクラス"""

    @staticmethod
    def scrape(race_id: str) -> pd.DataFrame:
        """出馬表をスクレイピング"""
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
                print("Shutuba table not found.")
                return pd.DataFrame()

            # レース情報取得
            race_name = ""
            race_data01 = ""
            race_data02 = ""
            
            r_name_elem = soup.find("h1", {"class": "RaceName"})
            if r_name_elem: race_name = r_name_elem.text.strip()
            
            r_data01_elem = soup.find("div", {"class": "RaceData01"})
            if r_data01_elem: race_data01 = r_data01_elem.text.strip()
                
            r_data02_elem = soup.find("div", {"class": "RaceData02"})
            if r_data02_elem: race_data02 = r_data02_elem.text.strip()

            race_info_text = race_data01 + " " + race_data02
            
            # コース情報の解析
            race_type = None
            course_len = None
            weather = None
            ground_state = None
            
            course_match = re.search(r"(芝|ダ|障)(\d+)m", race_info_text)
            if course_match:
                race_type = course_match.group(1)
                course_len = int(course_match.group(2))
            
            if "晴" in race_info_text: weather = "晴"
            elif "曇" in race_info_text: weather = "曇"
            elif "小雨" in race_info_text: weather = "小雨"
            elif "雨" in race_info_text: weather = "雨"
            elif "小雪" in race_info_text: weather = "小雪"
            elif "雪" in race_info_text: weather = "雪"
            
            if "不良" in race_info_text: ground_state = "不良"
            elif "稍重" in race_info_text: ground_state = "稍重"
            elif "重" in race_info_text: ground_state = "重"
            elif "良" in race_info_text: ground_state = "良"

            # 馬リスト
            horse_rows = table.find_all("tr", {"class": "HorseList"})
            data_list = []
            
            for row in horse_rows:
                waku_td = row.find("td", class_=re.compile(r"Waku"))
                waku = waku_td.text.strip() if waku_td else None
                
                umaban_td = row.find("td", class_=re.compile(r"Umaban"))
                umaban = umaban_td.text.strip() if umaban_td else None
                
                horse_td = row.find("td", {"class": "HorseInfo"})
                horse_name = None
                horse_id = None
                if horse_td:
                    h_link = horse_td.find("a")
                    if h_link:
                        horse_name = h_link.text.strip()
                        if "href" in h_link.attrs:
                            found = re.search(r"/horse/(\d+)", h_link["href"])
                            if found: horse_id = found.group(1)
                
                sex_age_td = row.find("td", {"class": "Barei"})
                sex_age = sex_age_td.text.strip() if sex_age_td else None
                
                cols = row.find_all("td")
                weight = cols[5].text.strip() if len(cols) > 5 else None
                
                jockey_td = row.find("td", {"class": "Jockey"})
                jockey_name = None
                jockey_id = None
                if jockey_td:
                    j_link = jockey_td.find("a")
                    if j_link:
                        jockey_name = j_link.text.strip()
                        if "href" in j_link.attrs:
                            found = re.search(r"/jockey/(\d+)", j_link["href"])
                            if found: jockey_id = found.group(1)
                
                h_weight_td = row.find("td", {"class": "Weight"})
                h_weight = h_weight_td.text.strip() if h_weight_td else None
                
                odds = None
                pop = None
                odds_elem = row.find("span", {"class": "Odds_Ninki"})
                if odds_elem: odds = odds_elem.text.strip()
                pop_elem = row.find("span", {"class": "Popular_Ninki"}) or row.find("td", {"class": "Popular_Ninki"})
                if pop_elem: pop = pop_elem.text.strip()
                
                data_list.append({
                    "枠番": waku, "馬番": umaban, "馬名": horse_name, "horse_id": horse_id,
                    "性齢": sex_age, "斤量": weight, "騎手": jockey_name, "jockey_id": jockey_id,
                    "馬体重": h_weight, "単勝": odds, "人気": pop,
                    "race_type": race_type, "course_len": course_len, "weather": weather, "ground_state": ground_state
                })
            
            # APIからオッズを取得
            try:
                api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&compress=0&output=json"
                api_res = requests.get(api_url, headers=HEADERS, timeout=15)
                api_res.encoding = "EUC-JP"
                odds_data = api_res.json()
                
                # status='middle'（レース前）と status='result'（レース後）の両方に対応
                if odds_data.get("status") in ["middle", "result"]:
                    if "data" in odds_data and odds_data["data"]:
                        data_content = odds_data["data"]
                        if isinstance(data_content, dict) and "odds" in data_content:
                            all_odds = data_content["odds"]
                            raw_odds = all_odds.get("1", {})  # 単勝オッズ
                            
                            for row_data in data_list:
                                u_num = str(row_data["馬番"])
                                # API keys are zero-padded (e.g., "01", "02")
                                u_num_padded = u_num.zfill(2)
                                
                                # Check both original and padded keys
                                target_key = None
                                if u_num in raw_odds: target_key = u_num
                                elif u_num_padded in raw_odds: target_key = u_num_padded
                                
                                if target_key and raw_odds[target_key]:
                                    if len(raw_odds[target_key]) > 0:
                                        row_data["単勝"] = raw_odds[target_key][0]
                                    if len(raw_odds[target_key]) > 2:
                                        row_data["人気"] = raw_odds[target_key][2]
            except Exception as e:
                print(f"[DEBUG] Error fetching odds from API: {e}")

            needs_fallback = any(not r.get("単勝") or "---" in str(r.get("単勝")) for r in data_list)
            
            if needs_fallback:
                try:
                    yoso_url = f"https://race.netkeiba.com/yoso/mark_list.html?race_id={race_id}"
                    y_res = requests.get(yoso_url, headers=HEADERS)
                    y_res.encoding = "EUC-JP"
                    y_soup = BeautifulSoup(y_res.text, "lxml")
                    y_wrap = y_soup.find("div", {"class": "YosoTableWrap"})
                    if y_wrap:
                        dls = y_wrap.find_all("dl", recursive=False)
                        y_names, y_odds, y_pops = [], [], []
                        for dl in dls:
                            title = dl.find("dt").text.strip() if dl.find("dt") else ""
                            items = [li.text.strip() for li in dl.find_all("li")]
                            if "馬名" in title: y_names = items
                            elif "単勝オッズ" in title: y_odds = items
                            elif "人気" in title: y_pops = items
                        if y_names:
                            for row_data in data_list:
                                if row_data.get("馬名") in y_names:
                                    idx = y_names.index(row_data["馬名"])
                                    if not row_data.get("単勝") or "---" in str(row_data.get("単勝")):
                                        if idx < len(y_odds): row_data["単勝"] = y_odds[idx]
                                    if not row_data.get("人気") or "**" in str(row_data.get("人気")):
                                        if idx < len(y_pops): row_data["人気"] = y_pops[idx]
                except: pass

            df_final = pd.DataFrame(data_list)
            df_final['レース名'] = race_name # カラムとして追加
            df_final.index = [race_id] * len(df_final)
            df_final.attrs.update({'race_name': race_name, 'race_data01': race_data01, 'race_data02': race_data02})
            return df_final
            
        except Exception as e:
            print(f"Error scraping shutuba {race_id}: {e}")
            return pd.DataFrame()

class Odds:
    """リアルタイムオッズスクレイピングクラス"""
    
    # オッズページのタイプ定義
    TYPES = {
        'tan': 'b1',   # 単勝・複勝
        'waku': 'b3',  # 枠連
        'umaren': 'b4',# 馬連
        'wide': 'b5',  # ワイド
        'umatan': 'b6',# 馬単
        'sanrenpuku': 'b7', # 3連複
        'sanrentan': 'b8'   # 3連単
    }

    @staticmethod
    def scrape(race_id: str) -> dict:
        """
        全券種のオッズを取得して辞書形式で返す(API使用)
        
        Args:
            race_id: レースID
            
        Returns:
            dict: {
                'tan': {umaban: odds, ...},
                'fuku': {umaban: [min_odds, max_odds], ...},
                'umaren': {(uma1, uma2): odds, ...},
                'wide': {(uma1, uma2): [min_odds, max_odds], ...},
                'umatan': {(uma1, uma2): odds, ...},
                'sanrenpuku': {(uma1, uma2, uma3): odds, ...},
                'sanrentan': {(uma1, uma2, uma3): odds, ...}
            }
        """
        try:
            odds_data = {
                'tan': {},
                'fuku': {},
                'umaren': {},
                'wide': {},
                'umatan': {},
                'sanrenpuku': {},
                'sanrentan': {}
            }
            
            # JRA API (全券種を取得)
            # 重要: action=init と compress=0 が必須パラメータ
            # type=all で全ての券種を一度に取得
            api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=all&action=init&compress=0&output=json"
            print(f"[ODDS DEBUG] Fetching API: {api_url}")
            
            response = requests.get(api_url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            api_data = response.json()
            
            print(f"[ODDS DEBUG] API status: {api_data.get('status')}")
            print(f"[ODDS DEBUG] API reason: {api_data.get('reason')}")
            
            # status='middle' は正常な状態(レース確定前)
            # dataフィールドに実際のオッズが格納されている
            if 'data' in api_data and api_data['data']:
                data_content = api_data['data']
                
                if isinstance(data_content, dict) and 'odds' in data_content:
                    all_odds = data_content['odds']
                    
                    # 単勝オッズ (type=1)
                    tan_odds = all_odds.get('1', {})
                    print(f"[ODDS DEBUG] Raw TAN odds count: {len(tan_odds)}")
                    
                    for umaban_str, odds_info in tan_odds.items():
                        try:
                            umaban = int(umaban_str)
                            # odds_info is a list: [odds, favorite_mark, popularity]
                            if odds_info and len(odds_info) > 0:
                                odds_value = float(odds_info[0])
                                if odds_value > 0:
                                    odds_data['tan'][umaban] = odds_value
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG TAN] Error parsing umaban={umaban_str}: {e}")
                            continue
                    
                    # 複勝オッズ (type=2)
                    fuku_odds = all_odds.get('2', {})
                    print(f"[ODDS DEBUG] Raw FUKU odds count: {len(fuku_odds)}")
                    
                    for umaban_str, odds_info in fuku_odds.items():
                        try:
                            umaban = int(umaban_str)
                            # odds_info is a list: [min_odds, max_odds, popularity]
                            if odds_info and len(odds_info) >= 2:
                                min_odds = float(odds_info[0])
                                max_odds = float(odds_info[1])
                                if min_odds > 0 and max_odds > 0:
                                    odds_data['fuku'][umaban] = [min_odds, max_odds]
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG FUKU] Error parsing umaban={umaban_str}: {e}")
                            continue
                    
                    # 馬連オッズ (type=4)
                    umaren_odds = all_odds.get('4', {})
                    print(f"[ODDS DEBUG] Raw UMAREN odds count: {len(umaren_odds)}")
                    
                    for combination_str, odds_info in umaren_odds.items():
                        try:
                            # combination_str is like "0102" (no hyphen, zero-padded)
                            if len(combination_str) == 4:
                                uma1 = int(combination_str[:2])
                                uma2 = int(combination_str[2:])
                                if odds_info and len(odds_info) > 0:
                                    # Remove comma from odds value
                                    odds_value = float(odds_info[0].replace(',', ''))
                                    if odds_value > 0:
                                        odds_data['umaren'][(uma1, uma2)] = odds_value
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG UMAREN] Error parsing {combination_str}: {e}")
                            continue
                    
                    # ワイドオッズ (type=5)
                    wide_odds = all_odds.get('5', {})
                    print(f"[ODDS DEBUG] Raw WIDE odds count: {len(wide_odds)}")
                    
                    for combination_str, odds_info in wide_odds.items():
                        try:
                            # combination_str is like "0102" (no hyphen, zero-padded)
                            if len(combination_str) == 4:
                                uma1 = int(combination_str[:2])
                                uma2 = int(combination_str[2:])
                                if odds_info and len(odds_info) >= 2:
                                    min_odds = float(odds_info[0].replace(',', ''))
                                    max_odds = float(odds_info[1].replace(',', ''))
                                    if min_odds > 0 and max_odds > 0:
                                        odds_data['wide'][(uma1, uma2)] = [min_odds, max_odds]
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG WIDE] Error parsing {combination_str}: {e}")
                            continue
                    
                    # 馬単オッズ (type=6)
                    umatan_odds = all_odds.get('6', {})
                    print(f"[ODDS DEBUG] Raw UMATAN odds count: {len(umatan_odds)}")
                    
                    for combination_str, odds_info in umatan_odds.items():
                        try:
                            # combination_str is like "0102" (1着→2着の順序指定)
                            if len(combination_str) == 4:
                                uma1 = int(combination_str[:2])
                                uma2 = int(combination_str[2:])
                                if odds_info and len(odds_info) > 0:
                                    odds_value = float(odds_info[0].replace(',', ''))
                                    if odds_value > 0:
                                        odds_data['umatan'][(uma1, uma2)] = odds_value
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG UMATAN] Error parsing {combination_str}: {e}")
                            continue
                    
                    # 三連複オッズ (type=7)
                    sanrenpuku_odds = all_odds.get('7', {})
                    print(f"[ODDS DEBUG] Raw SANRENPUKU odds count: {len(sanrenpuku_odds)}")
                    
                    for combination_str, odds_info in sanrenpuku_odds.items():
                        try:
                            # combination_str is like "010203" (no hyphen, zero-padded)
                            if len(combination_str) == 6:
                                uma1 = int(combination_str[:2])
                                uma2 = int(combination_str[2:4])
                                uma3 = int(combination_str[4:])
                                if odds_info and len(odds_info) > 0:
                                    odds_value = float(odds_info[0].replace(',', ''))
                                    if odds_value > 0:
                                        odds_data['sanrenpuku'][(uma1, uma2, uma3)] = odds_value
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG SANRENPUKU] Error parsing {combination_str}: {e}")
                            continue
                    
                    # 三連単オッズ (type=8)
                    sanrentan_odds = all_odds.get('8', {})
                    print(f"[ODDS DEBUG] Raw SANRENTAN odds count: {len(sanrentan_odds)}")
                    
                    for combination_str, odds_info in sanrentan_odds.items():
                        try:
                            # combination_str is like "010203" (1着→2着→3着の順序指定)
                            if len(combination_str) == 6:
                                uma1 = int(combination_str[:2])
                                uma2 = int(combination_str[2:4])
                                uma3 = int(combination_str[4:])
                                if odds_info and len(odds_info) > 0:
                                    # Remove comma from large odds values like "1,180.0"
                                    odds_value = float(odds_info[0].replace(',', ''))
                                    if odds_value > 0:
                                        odds_data['sanrentan'][(uma1, uma2, uma3)] = odds_value
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG SANRENTAN] Error parsing {combination_str}: {e}")
                            continue
                else:
                    print(f"[ODDS DEBUG] Data field exists but odds structure is unexpected")
            else:
                print(f"[ODDS DEBUG] No data field or data is empty")
            
            print(f"[ODDS DEBUG] Final extracted odds counts:")
            print(f"  - TAN: {len(odds_data['tan'])}")
            print(f"  - FUKU: {len(odds_data['fuku'])}")
            print(f"  - UMAREN: {len(odds_data['umaren'])}")
            print(f"  - WIDE: {len(odds_data['wide'])}")
            print(f"  - UMATAN: {len(odds_data['umatan'])}")
            print(f"  - SANRENPUKU: {len(odds_data['sanrenpuku'])}")
            print(f"  - SANRENTAN: {len(odds_data['sanrentan'])}")
            
            # オッズが全て空の場合の警告
            if not any([odds_data['tan'], odds_data['fuku'], odds_data['umaren'], 
                       odds_data['wide'], odds_data['umatan'], odds_data['sanrenpuku'], 
                       odds_data['sanrentan']]):
                print(f"[ODDS WARNING] オッズデータが取得できませんでした。")
                print(f"[ODDS WARNING] このレースはオッズ発売前か、既に締め切られている可能性があります。")
                print(f"[ODDS WARNING] 現在発売中のレース(未来のレース)のURLをお試しください。")
            
            return odds_data

        except Exception as e:
            import traceback
            print(f"Error scraping odds for {race_id}: {e}")
            traceback.print_exc()
            return {
                'tan': {}, 'fuku': {}, 'umaren': {}, 'wide': {}, 
                'umatan': {}, 'sanrenpuku': {}, 'sanrentan': {}
            }
        """
        全券種のオッズを取得して辞書形式で返す(API使用)
        
        Args:
            race_id: レースID
            
        Returns:
            dict: {
                'tan': {umaban: odds, ...},
                'fuku': {umaban: [min_odds, max_odds], ...},
            }
        """
        try:
            odds_data = {
                'tan': {},
                'fuku': {}
            }
            
            # JRA API (単勝・複勝両方を取得)
            # 重要: action=init と compress=0 が必須パラメータ
            api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&compress=0&output=json"
            print(f"[ODDS DEBUG] Fetching API: {api_url}")
            
            response = requests.get(api_url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            api_data = response.json()
            
            print(f"[ODDS DEBUG] API status: {api_data.get('status')}")
            print(f"[ODDS DEBUG] API reason: {api_data.get('reason')}")
            
            # status='middle' は正常な状態(レース確定前)
            # dataフィールドに実際のオッズが格納されている
            if 'data' in api_data and api_data['data']:
                data_content = api_data['data']
                
                if isinstance(data_content, dict) and 'odds' in data_content:
                    all_odds = data_content['odds']
                    
                    # 単勝オッズ (type=1)
                    tan_odds = all_odds.get('1', {})
                    print(f"[ODDS DEBUG] Raw TAN odds count: {len(tan_odds)}")
                    
                    for umaban_str, odds_info in tan_odds.items():
                        try:
                            umaban = int(umaban_str)
                            # odds_info is a list: [odds, favorite_mark, popularity]
                            if odds_info and len(odds_info) > 0:
                                odds_value = float(odds_info[0])
                                if odds_value > 0:
                                    odds_data['tan'][umaban] = odds_value
                                    print(f"[ODDS DEBUG TAN] Added tan[{umaban}] = {odds_value}")
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG TAN] Error parsing umaban={umaban_str}: {e}")
                            continue
                    
                    # 複勝オッズ (type=2)
                    fuku_odds = all_odds.get('2', {})
                    print(f"[ODDS DEBUG] Raw FUKU odds count: {len(fuku_odds)}")
                    
                    for umaban_str, odds_info in fuku_odds.items():
                        try:
                            umaban = int(umaban_str)
                            # odds_info is a list: [min_odds, max_odds, popularity]
                            if odds_info and len(odds_info) >= 2:
                                min_odds = float(odds_info[0])
                                max_odds = float(odds_info[1])
                                if min_odds > 0 and max_odds > 0:
                                    odds_data['fuku'][umaban] = [min_odds, max_odds]
                                    print(f"[ODDS DEBUG FUKU] Added fuku[{umaban}] = [{min_odds}, {max_odds}]")
                        except (ValueError, IndexError, TypeError) as e:
                            print(f"[ODDS DEBUG FUKU] Error parsing umaban={umaban_str}: {e}")
                            continue
                else:
                    print(f"[ODDS DEBUG] Data field exists but odds structure is unexpected")
            else:
                print(f"[ODDS DEBUG] No data field or data is empty")
            
            print(f"[ODDS DEBUG] Final extracted tan odds count: {len(odds_data['tan'])}")
            print(f"[ODDS DEBUG] Final extracted fuku odds count: {len(odds_data['fuku'])}")
            
            # オッズが全て空の場合の警告
            if not odds_data['tan'] and not odds_data['fuku']:
                print(f"[ODDS WARNING] オッズデータが取得できませんでした。")
                print(f"[ODDS WARNING] このレースはオッズ発売前か、既に締め切られている可能性があります。")
                print(f"[ODDS WARNING] 現在発売中のレース(未来のレース)のURLをお試しください。")
            
            return odds_data

        except Exception as e:
            import traceback
            print(f"Error scraping odds for {race_id}: {e}")
            traceback.print_exc()
            return {'tan': {}, 'fuku': {}}

        """
        全券種のオッズを取得して辞書形式で返す
        
        Args:
            race_id: レースID
            
        Returns:
            dict: {
                'tan': {umaban: odds, ...},
                'fuku': {umaban: [min_odds, max_odds], ...},
            }
        """
        try:
            url = f"https://race.netkeiba.com/odds/index.html?type={Odds.TYPES['tan']}&race_id={race_id}"
            print(f"[ODDS DEBUG] Fetching URL: {url}")
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.encoding = "EUC-JP"
            
            if response.status_code != 200:
                print(f"[ODDS DEBUG] Failed to fetch odds page: status {response.status_code}")
                return {'tan': {}, 'fuku': {}}
            
            soup = BeautifulSoup(response.text, "lxml")
            
            odds_data = {
                'tan': {},
                'fuku': {}
            }
            
            # 単勝オッズの取得 (div#odds_tan_block内のテーブル)
            tan_block = soup.find("div", {"id": "odds_tan_block"})
            if tan_block:
                tan_table = tan_block.find("table", class_="RaceOdds_HorseList_Table")
                if tan_table:
                    rows = tan_table.find_all("tr")[1:]  # ヘッダー行をスキップ
                    print(f"[ODDS DEBUG] Found {len(rows)} rows in tan table")
                    
                    for row in rows:
                        try:
                            cells = row.find_all("td")
                            if len(cells) < 6:
                                print(f"[ODDS DEBUG TAN] Row has only {len(cells)} cells, skipping")
                                continue
                            
                            # 馬番は2番目のtd (index 1)
                            umaban_td = cells[1]
                            umaban_text = umaban_td.text.strip()
                            print(f"[ODDS DEBUG TAN] Cell[1] text: '{umaban_text}'")
                            
                            if not umaban_text or not umaban_text.isdigit():
                                print(f"[ODDS DEBUG TAN] Skipping row - umaban not digit")
                                continue
                            umaban = int(umaban_text)
                            
                            # オッズは6番目のtd内のspan.Odds (index 5)
                            odds_td = cells[5]
                            print(f"[ODDS DEBUG TAN] Cell[5] text: '{odds_td.text.strip()}', html: {odds_td}[:100]")
                            
                            odds_span = odds_td.find("span", class_="Odds")
                            if odds_span:
                                odds_text = odds_span.text.strip()
                                print(f"[ODDS DEBUG TAN] Found span.Odds with text: '{odds_text}'")
                                # Check if odds_text is valid (not empty, not starting with '---')
                                if odds_text and not odds_text.startswith('---'):
                                    try:
                                        odds_data['tan'][umaban] = float(odds_text)
                                        print(f"[ODDS DEBUG TAN] Successfully added tan[{umaban}] = {odds_text}")
                                    except ValueError as ve:
                                        print(f"[ODDS DEBUG TAN] ValueError converting '{odds_text}': {ve}")
                            else:
                                print(f"[ODDS DEBUG TAN] No span.Odds found in cell[5]")
                        except Exception as row_err:
                            print(f"[ODDS DEBUG TAN] Row error: {row_err}")
                            continue
            else:
                print(f"[ODDS DEBUG] No odds_tan_block found")
            
            # 複勝オッズの取得 (div#odds_fuku_block内のテーブル)
            fuku_block = soup.find("div", {"id": "odds_fuku_block"})
            if fuku_block:
                fuku_table = fuku_block.find("table", class_="RaceOdds_HorseList_Table")
                if fuku_table:
                    rows = fuku_table.find_all("tr")[1:]  # ヘッダー行をスキップ
                    print(f"[ODDS DEBUG] Found {len(rows)} rows in fuku table")
                    
                    for row in rows:
                        try:
                            cells = row.find_all("td")
                            if len(cells) < 6:
                                continue
                            
                            # 馬番は2番目のtd (index 1)
                            umaban_td = cells[1]
                            umaban_text = umaban_td.text.strip()
                            if not umaban_text or not umaban_text.isdigit():
                                continue
                            umaban = int(umaban_text)
                            
                            # オッズは6番目のtd内のspan.Odds (index 5)
                            odds_td = cells[5]
                            odds_span = odds_td.find("span", class_="Odds")
                            if odds_span:
                                odds_text = odds_span.text.strip()
                                # Check if odds_text is valid (not empty, not starting with '---')
                                if odds_text and not odds_text.startswith('---'):
                                    # "9.0 - 15.1" のような形式
                                    try:
                                        parts = odds_text.split('-')
                                        if len(parts) == 2:
                                            min_odds = float(parts[0].strip())
                                            max_odds = float(parts[1].strip())
                                            odds_data['fuku'][umaban] = [min_odds, max_odds]
                                        elif len(parts) == 1:
                                            # 単一値の場合
                                            odds = float(parts[0].strip())
                                            odds_data['fuku'][umaban] = [odds, odds]
                                    except ValueError:
                                        pass
                        except Exception as row_err:
                            continue
            else:
                print(f"[ODDS DEBUG] No odds_fuku_block found")
            
            
            print(f"[ODDS DEBUG] Extracted tan odds: {odds_data['tan']}")
            print(f"[ODDS DEBUG] Extracted fuku odds: {odds_data['fuku']}")
            
            # オッズが全て空の場合の警告
            if not odds_data['tan'] and not odds_data['fuku']:
                print(f"[ODDS WARNING] オッズデータが取得できませんでした。")
                print(f"[ODDS WARNING] このレースはオッズ発売前か、既に締め切られている可能性があります。")
                print(f"[ODDS WARNING] 現在発売中のレース（未来のレース）のURLをお試しください。")
            
            return odds_data

        except Exception as e:
            import traceback
            print(f"Error scraping odds for {race_id}: {e}")
            traceback.print_exc()
            return {'tan': {}, 'fuku': {}}

