"""
日次予想レポート成績集計スクリプト (汎用版)

対象:
  - reports/prediction_{YYYYMMDD}_all_models.md
  - (存在すれば) reports/prediction_{YYYYMMDD}_catboost_ev_*.md

払戻データ: data/raw/payouts_{YYYYMMDD}.pkl
"""
import os
import sys
import pickle
import re
import argparse
import itertools
import pandas as pd
from tabulate import tabulate
import glob

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')


def load_payouts(path: str) -> dict:
    """払戻データを読み込む。"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_bets_all_models(report_path: str) -> dict:
    """all_modelsレポートからベットを抽出する。"""
    bets_by_race: dict = {}
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    race_blocks = re.split(r'\n## ', content)
    for block in race_blocks:
        block = block.strip()
        if not block:
            continue

        m_race = re.match(r'.*?\((\d{12})\)', block)
        if not m_race:
            continue
        race_id = m_race.group(1)
        if race_id not in bets_by_race:
            bets_by_race[race_id] = []

        current_model = None
        current_strategy = None
        current_confidence = None

        for line in block.split('\n'):
            line = line.strip()
            
            # モデル・戦略行
            m_ms = re.match(r'###\s+モデル:\s+(\S+)\s+/\s+戦略:\s+(\S+)\s+\(自信度:\s+(\S+)\)', line)
            if m_ms:
                current_model = m_ms.group(1)
                current_strategy = m_ms.group(2)
                current_confidence = m_ms.group(3)
                continue

            if not current_model or not current_strategy:
                continue

            # BOXパース
            m_box = re.match(r'-\s+BOX:\s+\[([^\]]+)\]\s+\((\d+)点\s+x\s+(\d+)円', line)
            if m_box:
                horses = [int(h.strip()) for h in m_box.group(1).split(',')]
                amount_per = int(m_box.group(3))
                if current_strategy == 'box4_sanrenpuku':
                    combs = [tuple(sorted(c)) for c in itertools.combinations(horses, 3)]
                    for c in combs:
                        bets_by_race[race_id].append({
                            'model': current_model, 'strategy': current_strategy,
                            'confidence': current_confidence, 'type_key': 'sanrenpuku',
                            'combination': c, 'amount': amount_per
                        })
                continue

            # 流しパース
            m_nagashi = re.match(r'-\s+流し:\s+\[([^\]]+)\]\s+\((\d+)点\s+x\s+(\d+)円', line)
            if m_nagashi:
                horses = [int(h.strip()) for h in m_nagashi.group(1).split(',')]
                amount_per = int(m_nagashi.group(3))
                if current_strategy == 'ranking_anchor':
                    axis, partners = horses[0], horses[1:]
                    combs = [tuple(sorted((axis, p1, p2))) for p1, p2 in itertools.combinations(partners, 2)]
                    for c in combs:
                        bets_by_race[race_id].append({
                            'model': current_model, 'strategy': current_strategy,
                            'confidence': current_confidence, 'type_key': 'sanrenpuku',
                            'combination': c, 'amount': amount_per
                        })
                elif current_strategy == 'wide_nagashi':
                    axis, partners = horses[0], horses[1:]
                    for p in partners:
                        bets_by_race[race_id].append({
                            'model': current_model, 'strategy': current_strategy,
                            'confidence': current_confidence, 'type_key': 'wide',
                            'combination': tuple(sorted((axis, p))), 'amount': amount_per
                        })
                continue

            # SINGLEパース (ranking_anchorの複勝扱いなど)
            m_single = re.match(r'-\s+SINGLE:\s+\[(\d+)\]\s+\((\d+)点\s+x\s+(\d+)円\s+=\s+計(\d+)円\)', line)
            if m_single:
                horse = int(m_single.group(1))
                amount = int(m_single.group(4))
                bets_by_race[race_id].append({
                    'model': current_model, 'strategy': current_strategy,
                    'confidence': current_confidence, 'type_key': 'fuku',
                    'combination': (horse,), 'amount': amount
                })
                continue
    return bets_by_race


def extract_bets_catboost(report_path: str) -> dict:
    """CatBoostレポートからベットを抽出する。"""
    bets_by_race: dict = {}
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    race_blocks = re.split(r'\n### ', content)
    for block in race_blocks:
        block = block.strip()
        if not block:
            continue

        m_race = re.match(r'.*?\((\d{12})\)', block)
        if not m_race:
            continue
        race_id = m_race.group(1)
        if race_id not in bets_by_race:
            bets_by_race[race_id] = []

        m_box = re.search(r'-\s+\*\*BOX\*\*:\s+\[([^\]]+)\]\s+\((\d+)点\s+x\s+(\d+)円', block)
        if m_box:
            horses = [int(h.strip()) for h in m_box.group(1).split(',')]
            amount_per = int(m_box.group(3))
            combs = [tuple(sorted(c)) for c in itertools.combinations(horses, 3)]
            for c in combs:
                bets_by_race[race_id].append({
                    'model': 'catboost', 'strategy': 'box4_sanrenpuku',
                    'confidence': '-', 'type_key': 'sanrenpuku',
                    'combination': c, 'amount': amount_per
                })
    return bets_by_race


def verify_bets(all_bets: list, payouts: dict) -> pd.DataFrame:
    """ベットリストと払戻データを照合して成績を集計する。"""
    metrics: dict = {}
    for bet in all_bets:
        race_id, model, strategy = bet['race_id'], bet['model'], bet['strategy']
        key = f"{model}/{strategy}"

        if key not in metrics:
            metrics[key] = {
                'model': model, 'strategy': strategy,
                'bet_amount': 0, 'return_amount': 0, 'hit_count': 0,
                'total_bets': 0, 'race_ids': set()
            }

        metrics[key]['bet_amount'] += bet['amount']
        metrics[key]['total_bets'] += 1
        metrics[key]['race_ids'].add(race_id)

        if race_id not in payouts:
            continue

        race_payout = payouts[race_id]
        type_key = bet['type_key']
        comb = bet['combination']

        if type_key not in race_payout:
            continue

        winning_combs = race_payout[type_key]
        hit = False
        payout_amt = 0

        if len(comb) == 1 and type_key in ['tan', 'fuku']:
            check_key = comb[0]
            if check_key in winning_combs:
                hit = True
                payout_amt = winning_combs[check_key]
        else:
            if comb in winning_combs:
                hit = True
                payout_amt = winning_combs[comb]

        if hit:
            ret = (payout_amt / 100) * bet['amount']
            metrics[key]['return_amount'] += ret
            metrics[key]['hit_count'] += 1

    rows = []
    for key, m in metrics.items():
        roi = (m['return_amount'] / m['bet_amount'] * 100) if m['bet_amount'] > 0 else 0
        hit_rate = (m['hit_count'] / m['total_bets'] * 100) if m['total_bets'] > 0 else 0
        rows.append({
            'モデル': m['model'], '戦略': m['strategy'],
            'レース数': len(m['race_ids']), '買い目数': m['total_bets'],
            '投資額': int(m['bet_amount']), '回収額': int(m['return_amount']),
            '収支': int(m['return_amount'] - m['bet_amount']),
            '回収率(%)': round(roi, 1), '的中数': m['hit_count'],
            '的中率(%)': round(hit_rate, 2)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('回収率(%)', ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='指定日の予想レポートの成績を集計')
    parser.add_argument('--date', required=True, help='対象日付 YYYYMMDD形式')
    args = parser.parse_args()
    date_str = args.date

    payout_file = os.path.join(PROJECT_ROOT, f"data/raw/payouts_{date_str}.pkl")
    all_models_report = os.path.join(PROJECT_ROOT, f"reports/prediction_{date_str}_all_models.md")
    
    # CatBoostレポートは時刻付きなど複数ある場合を考慮し最新を取得、無ければスキップ
    cat_reports = glob.glob(os.path.join(PROJECT_ROOT, f"reports/prediction_{date_str}_catboost_ev*.md"))
    catboost_report = sorted(cat_reports)[-1] if cat_reports else None

    if not os.path.exists(payout_file):
        print(f"エラー: 払戻データが見つかりません: {payout_file}")
        sys.exit(1)
        
    payouts = load_payouts(payout_file)
    print(f"払戻データ読み込み: {payout_file} ({len(payouts)}レース分)")

    all_combined = []

    # 1. all_models
    if os.path.exists(all_models_report):
        print(f"\n=== all_models 解析: {os.path.basename(all_models_report)} ===")
        bets_all = extract_bets_all_models(all_models_report)
        print(f"  → {len(bets_all)} レース抽出")
        
        flat_all = []
        for race_id, bets in bets_all.items():
            for b in bets:
                b['race_id'] = race_id
                b['source'] = 'all_models'
                flat_all.append(b)
        
        all_combined.extend(flat_all)
        
        df_all = verify_bets(flat_all, payouts)
        if not df_all.empty:
            print(tabulate(df_all, headers='keys', tablefmt='pipe', showindex=False))

    # 2. catboost_ev
    if catboost_report and os.path.exists(catboost_report):
        print(f"\n=== CatBoost 解析: {os.path.basename(catboost_report)} ===")
        bets_cat = extract_bets_catboost(catboost_report)
        print(f"  → {len(bets_cat)} レース抽出")
        
        flat_cat = []
        for race_id, bets in bets_cat.items():
            for b in bets:
                b['race_id'] = race_id
                b['source'] = 'catboost_ev'
                flat_cat.append(b)
        
        all_combined.extend(flat_cat)
        
        df_cat = verify_bets(flat_cat, payouts)
        if not df_cat.empty:
            print(tabulate(df_cat, headers='keys', tablefmt='pipe', showindex=False))

    # 全体サマリー
    if all_combined:
        print("\n" + "=" * 80)
        print("【全体サマリー】")
        print("=" * 80)
        df_total = verify_bets(all_combined, payouts)
        total_invest = df_total['投資額'].sum()
        total_return = df_total['回収額'].sum()
        total_net = total_return - total_invest
        total_roi = (total_return / total_invest * 100) if total_invest > 0 else 0
        total_hits = df_total['的中数'].sum()
        total_bet_count = df_total['買い目数'].sum()

        print(f"  総投資額: {total_invest:,}円")
        print(f"  総回収額: {total_return:,}円")
        print(f"  総収支:   {total_net:+,}円")
        print(f"  回収率:   {total_roi:.1f}%")
        print(f"  的中数:   {total_hits}/{total_bet_count}")

        # CSV出力
        out_csv = os.path.join(PROJECT_ROOT, f"reports/verify_result_{date_str}.csv")
        # source情報を付与しておく
        df_out = pd.DataFrame()
        if 'flat_all' in locals() and flat_all:
            d = df_all.copy()
            d['ソース'] = 'all_models'
            df_out = pd.concat([df_out, d])
        if 'flat_cat' in locals() and flat_cat:
            d = df_cat.copy()
            d['ソース'] = 'catboost_ev'
            df_out = pd.concat([df_out, d])
            
        if not df_out.empty:
            df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
            print(f"\nCSV出力: {out_csv}")
    else:
        print("集計対象のデータが見つかりませんでした。")

if __name__ == "__main__":
    main()
