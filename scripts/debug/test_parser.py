
import re

REPORT_FILE = "prediction_report_20260207_hybrid.md"

def test_parser():
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    regex_venue = re.compile(r"## (.+)開催")
    regex_race = re.compile(r"### (.+)(\d{2})R")
    regex_bet = re.compile(r"- \*\*(.+) (BOX|SINGLE|流し)\*\*: ([\d\-]+)(?: BOX)? \((\d+)円\)")
    
    current_venue = None
    current_race_num = None
    bets_count = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        m_venue = regex_venue.match(line)
        if m_venue:
            print(f"[{i+1}] Venue found: {m_venue.group(1)}")
            current_venue = m_venue.group(1)
            continue
            
        m_race = regex_race.match(line)
        if m_race:
            print(f"[{i+1}] Race found: {m_race.group(0)} -> {m_race.group(2)}R")
            current_race_num = m_race.group(2)
            continue
            
        m_bet = regex_bet.match(line)
        if m_bet:
            print(f"[{i+1}] Bet found: {m_bet.groups()} under Race {current_race_num}")
            bets_count += 1
        elif "- **" in line:
            print(f"[{i+1}] FAILED TO MATCH BET LINE: {line}")
            
    print(f"Total bets found: {bets_count}")

if __name__ == "__main__":
    test_parser()
