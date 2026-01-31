
import requests
import json

def test_reasoning():
    url = "http://127.0.0.1:5005/api/predict"
    
    # 33.0秒の上がりを持つ強い馬を作成（末脚評価が出ることを期待）
    # 騎手勝率も高くする（騎手評価が出ることを期待）
    # モデルが「勝ち」と予測しやすくするため、順位や勝率を最強にする
    horses = [
        {
            "馬番": 1, "馬名": "スピードスター", 
            "avg_last_3f": 33.0, "jockey_win_rate": 0.30, 
            "avg_rank": 1.0, "win_rate": 0.8, "place_rate": 1.0,
            "単勝": 2.0, "人気": 1, "race_count": 10
        },
        {
            "馬番": 2, "馬名": "普通の馬", 
            "avg_last_3f": 38.0, "jockey_win_rate": 0.05,
            "avg_rank": 8.0, "win_rate": 0.0,
            "単勝": 50.0, "人気": 10
        },
        {
            "馬番": 3, "馬名": "馬3", "avg_last_3f": 36.0, "jockey_win_rate": 0.1, "avg_rank": 3.0, "win_rate": 0.3, "単勝": 10.0, "人気": 3, "race_count": 10
        },
        {
            "馬番": 4, "馬名": "馬4", "avg_last_3f": 36.0, "jockey_win_rate": 0.1, "avg_rank": 4.0, "win_rate": 0.2, "単勝": 20.0, "人気": 4, "race_count": 10
        },
        {
            "馬番": 5, "馬名": "馬5", "avg_last_3f": 36.0, "jockey_win_rate": 0.1, "avg_rank": 5.0, "win_rate": 0.1, "単勝": 30.0, "人気": 5, "race_count": 10
        },
    ]
    
    payload = {
        "horses": horses,
        "budget": 10000
    }
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        print("Full Response Keys:", data.keys())
        # print("Full Response Data:", json.dumps(data, indent=2, ensure_ascii=False))
        
        print("Status Code:", response.status_code)
        
        if 'predictions' in data and data['predictions']:
            print("First few predictions:")
            for res in data['predictions'][:2]:
                print(f"Horse: {res['horse_name']}, Prob: {res['probability']:.4f}, EV: {res['expected_value']:.4f}")

        if 'recommendations' in data:
            print(f"Recommendations Count: {len(data['recommendations'])}")
            for rec in data['recommendations']:
                # キー名は bet_type, combination, etc.
                bet_type = rec.get('bet_type', 'Unknown')
                comb = rec.get('combination', 'Unknown')
                reason = rec.get('reason', 'No Reason')
                print(f"[{bet_type}] {comb} -> {reason}")
                
                # Check for keywords
                if "末脚" in reason:
                    print("SUCCESS: Reason contains '末脚'")
                if "名手" in reason:
                    print("SUCCESS: Reason contains '名手'")
                    
        else:
            print("No recommendations found.")
            print(data)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_reasoning()
