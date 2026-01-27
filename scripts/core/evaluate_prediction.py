"""
2026/01/25 レース予測結果の検証スクリプト
- 予測CSV (prediction_20260125.csv) を読み込み
- ネット競馬から実際の結果と払い戻しデータをスクレイピング
- 回収率を計算して表示
"""
import sys
import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm

# モジュールパス
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.constants import HEADERS

def get_race_result(race_id):
    """レース結果と払い戻し情報を取得"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = 'EUC-JP'
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 1. 着順テーブル
        # 最新のNetkeiba (result.html) は RaceTable01
        table = soup.find("table", {"class": "RaceTable01"})
        if not table:
            # 旧形式などの保険
            table = soup.find("table", {"class": "ResultsByRaceDetail"})
        
        if not table: return None
        
        df = pd.read_html(str(table))[0]
        # カラム名に改行が入っている場合があるので整理
        df.columns = [str(c).replace('\n', '') for c in df.columns]
        
        # 馬名からスペース削除
        if '馬名' in df.columns:
            df['馬名'] = df['馬名'].astype(str).str.strip()
            
        # 2. 払い戻しテーブル
        payouts = {}
        # class="Payout_Detail_Table" が複数ある
        payout_tables = soup.find_all("table", {"class": "Payout_Detail_Table"})
        
        # まだ見つからない場合 (class="Payout" の可能性)
        if not payout_tables:
            payout_tables = soup.find_all("table", {"class": "Payout"})
        
        # 単勝, 複勝, 枠連, 馬連, ワイド, 馬単, 3連複, 3連単
        # 構造が複雑なので簡易解析
        def clean_money(s):
            return int(str(s).replace(',', '').replace('円', ''))

        for pt in payout_tables:
            rows = pt.find_all('tr')
            for row in rows:
                th = row.find('th')
                if not th: continue
                kind = th.text.strip() # "単勝", "馬連" etc
                
                tds = row.find_all('td')
                if not tds: continue
                
                # 複数の払い戻しがある場合（ワイド、複勝）、改行やbrなどが含まれる
                # 簡易的に [Payout] = [(番, 配当), (番, 配当)...] とする
                
                # 馬番データの抽出
                if 'PayoutNums' in tds[0].attrs.get('class', []):
                    nums_raw = tds[0].decode_contents().split('<br/>') 
                    # BeautifulSoupのtextだとくっつくのでdecode_contentsか、get_text(separator='|')
                    nums_list = tds[0].get_text(separator='|').split('|')
                else:
                    nums_list = [tds[0].text.strip()]
                    
                # 金額データの抽出
                if 'PayoutMoney' in tds[1].attrs.get('class', []):
                    money_list = tds[1].get_text(separator='|').split('|')
                else:
                    money_list = [tds[1].text.strip()]
                
                # ペアにする
                # データ個数が合わない場合は調整（同着など）
                # ここでは簡易実装としてリスト化
                items = []
                for n, m in zip(nums_list, money_list):
                    n = n.strip()
                    m = m.strip()
                    if not n or not m: continue
                    try:
                        val = clean_money(m)
                        # 馬番は "1" とか "1-2" とか
                        items.append({'nums': n, 'pay': val})
                    except:
                        pass
                
                if kind in payouts:
                    payouts[kind].extend(items)
                else:
                    payouts[kind] = items
                    
        return {
            'df': df,
            'payouts': payouts
        }
    except Exception as e:
        print(f"Error getting result for {race_id}: {e}")
        return None

def evaluate():
    print("=== 2026/01/25 回収率検証 ===")
    
    # 1. 予測データの読み込み
    csv_path = 'prediction_20260125.csv'
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found.")
        return
    pred_df = pd.read_csv(csv_path)
    print(f"Prediction rows: {len(pred_df)}")
    
    # 2. 結果データの準備
    # 2026/01/25 の全レースID
    # 中山(06), 京都(08), 小倉(10)
    target_date_id = "20260125" # これは使わない
    # 1回中山9日 -> 2026060109
    # 1回京都9日 -> 2026080109
    # 1回小倉2日 -> 2026100102
    
    base_ids = ['2026060109', '2026080109', '2026100102']
    race_ids = [f"{b}{r:02d}" for b in base_ids for r in range(1, 13)]
    
    print("\nfetching actual results...")
    results_map = {} # race_id -> result_obj
    horse_race_map = {} # horse_name -> race_id
    
    for rid in tqdm(race_ids):
        time.sleep(1)
        res = get_race_result(rid)
        if res:
            results_map[rid] = res
            # 馬名マッピング作成
            r_df = res['df']
            if '馬名' in r_df.columns:
                for hname in r_df['馬名'].values:
                    horse_race_map[hname] = rid
    
    print(f"Results fetched: {len(results_map)} races")
    
    # 3. 予測データに正しい race_id を付与
    print("Matching predictions to results...")
    pred_df['true_race_id'] = pred_df['馬名'].map(horse_race_map)
    
    # マッチしなかった馬（除外や競走中止、あるいは名前不一致）
    missing = pred_df[pred_df['true_race_id'].isna()]
    if not missing.empty:
        print(f"Warning: {len(missing)} horses could not be matched to race results.")
        # print(missing['馬名'].unique())
        
    # race_id ごとに処理
    total_invest = 0
    total_return = 0
    
    details = []
    
    # Groupping by true_race_id
    valid_df = pred_df.dropna(subset=['true_race_id'])
    
    for rid, group in valid_df.groupby('true_race_id'):
        if rid not in results_map: continue
        
        # 予測スコア順にソート（predict_tomorrow.pyと同じロジック）
        group = group.sort_values('score', ascending=False)
        
        top1 = group.iloc[0]
        top2 = group.iloc[1] if len(group) > 1 else top1
        top3 = group.iloc[2] if len(group) > 2 else top2
        
        # ロジック再現
        score_diff = top1['score'] - top2['score']
        max_score = top1['score']
        
        # ロジック再現 (厳格化版)
        score_diff = top1['score'] - top2['score']
        max_score = top1['score']
        
        confidence = 'C'
        strategy = 'skip'
        
        if max_score >= 0.40 and score_diff >= 0.10:
            confidence = 'S'; strategy = 'winner'
        elif max_score >= 0.35:
            confidence = 'A'; strategy = 'standard'
        elif max_score >= 0.28:
            confidence = 'B'; strategy = 'balance'
        else:
            confidence = 'C'; strategy = 'skip'
            
        if strategy == 'skip':
            pass # 投資なし
        
        # 買い目リスト初期化 (必ず定義する)
        bets = []
        
        if strategy != 'skip':
            u1 = int(top1['馬番'])
            u2 = int(top2['馬番'])
            u3 = int(top3['馬番'])
            
            if strategy == 'winner':
                bets.append(('単勝', f"{u1}", 2500))
                bets.append(('馬連', f"{min(u1,u2)}-{max(u1,u2)}", 1500))
                bets.append(('ワイド', f"{min(u1,u3)}-{max(u1,u3)}", 1000))
            elif strategy == 'standard':
                bets.append(('単勝', f"{u1}", 1000))
                bets.append(('複勝', f"{u1}", 1500))
                bets.append(('馬連', f"{min(u1,u2)}-{max(u1,u2)}", 1000))
                bets.append(('ワイド', f"{min(u1,u2)}-{max(u1,u2)}", 1000))
                bets.append(('ワイド', f"{min(u1,u3)}-{max(u1,u3)}", 500))
            elif strategy == 'balance': 
                bets.append(('単勝', f"{u1}", 1000))
                bets.append(('馬連', f"{min(u1,u2)}-{max(u1,u2)}", 1000))
                bets.append(('馬連', f"{min(u1,u3)}-{max(u1,u3)}", 500))
                bets.append(('ワイド', f"{min(u1,u2)}-{max(u1,u2)}", 1500))
                bets.append(('ワイド', f"{min(u1,u3)}-{max(u1,u3)}", 1000))
            
        # 収支計算
        race_invest = 0
        race_return = 0
        payouts = results_map[rid]['payouts']
        
        matched_bets_log = []
        
        for kind, nums, amount in bets:
            race_invest += amount
            
            # 的中判定
            hit_pay = 0
            if kind in payouts:
                for p in payouts[kind]:
                    # payoutのnumsは "1" とか "1 - 2" (スペースありなし注意)
                    # Netkeibaは "1 - 2" のようにスペースが入ったり入らなかったり
                    # Normalize
                    p_nums = p['nums'].replace(' ', '').replace('-', '')
                    my_nums = nums.replace('-', '')
                    
                    # 単勝・複勝は1つの数字、馬連ワイドは2つの数字
                    # 順序を気にする必要があるか？ 馬連は昇順にしたのでOK。ワイドも。
                    
                    if p_nums == my_nums:
                        hit_pay += p['pay'] * (amount / 100)
            
            if hit_pay > 0:
                race_return += hit_pay
                matched_bets_log.append(f"{kind} {nums}: {int(hit_pay)}円")
        
        total_invest += race_invest
        total_return += race_return
        
        profit = race_return - race_invest
        details.append({
            'rid': rid,
            'name': group.iloc[0].get('馬名', 'Unknown'), # 代表馬
            'invest': race_invest,
            'return': race_return,
            'profit': profit,
            'hits': matched_bets_log
        })
        
    # 結果表示
    print("\n=== 検証結果 ===")
    print(f"総投資額: {total_invest:,}円")
    print(f"総回収額: {int(total_return):,}円")
    rate = (total_return / total_invest * 100) if total_invest > 0 else 0
    print(f"回収率: {rate:.1f}%")
    
    print("\n--- レース別収支 (上位5) ---")
    details.sort(key=lambda x: x['profit'], reverse=True)
    for d in details[:5]:
        print(f"ID:{d['rid']} 利益:{d['profit']:,}円 (投{d['invest']} 回{int(d['return'])}) {' '.join(d['hits'])}")

    print("\n--- ワースト3 ---")
    for d in details[-3:]:
        print(f"ID:{d['rid']} 利益:{d['profit']:,}円 (投{d['invest']} 回{int(d['return'])})")

if __name__ == '__main__':
    evaluate()
