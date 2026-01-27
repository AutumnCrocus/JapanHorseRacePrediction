# 競馬予想AI 定数定義

# netkeiba.comのベースURL
BASE_URL = "https://db.sp.netkeiba.com"
RACE_URL = BASE_URL + "/race/"
HORSE_URL = BASE_URL + "/horse/"
HORSE_PED_URL = BASE_URL + "/horse/ped/"
SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id="

# HTTPリクエスト設定
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_INTERVAL = 1  # リクエスト間隔（秒）
MAX_WORKERS = 10      # 並列実行数


import os

# プロジェクトルートディレクトリ（このファイルの2つ上のディレクトリ）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# データ保存ディレクトリ (絶対パス)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ファイル名
RESULTS_FILE = "results.pickle"
HORSE_RESULTS_FILE = "horse_results.pickle"
PEDS_FILE = "peds.pickle"
RETURN_FILE = "return_tables.pickle"

# 競馬場コード
PLACE_DICT = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉'
}

# レースタイプ
RACE_TYPE_DICT = {
    '芝': 'turf',
    'ダ': 'dirt',
    '障': 'obstacle'
}

# 天気
WEATHER_DICT = {
    '晴': 'sunny', '曇': 'cloudy', '小雨': 'light_rain',
    '雨': 'rain', '小雪': 'light_snow', '雪': 'snow'
}

# 馬場状態
GROUND_STATE_DICT = {
    '良': 'good', '稍重': 'slightly_heavy',
    '重': 'heavy', '不良': 'bad'
}

# 性別マッピング
SEX_DICT = {'牡': 0, '牝': 1, 'セ': 2}

# その他カテゴリ項目の数値化マップ (LabelEncoderの代わり、または補完)
RACE_TYPE_MAP = {'芝': 0, 'ダ': 1, '障': 2}
WEATHER_MAP = {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '小雪': 4, '雪': 5}
GROUND_MAP = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
