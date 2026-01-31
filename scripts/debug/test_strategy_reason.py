
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.strategy import BettingStrategy

def test_generate_reason():
    # Case 1: Zero Odds/EV for Sanrentan (High Payout)
    reason = BettingStrategy.generate_reason(
        'sanrentan', ['1', '2', '3'], 
        prob=0.1, ev=0.0, odds=0.0, features_list=[]
    )
    print(f"[Sanrentan 0.0] {reason}")
    assert "0.0倍" not in reason and "0.00倍" not in reason
    assert "一撃高配当狙い" in reason

    # Case 2: Zero EV for Umaren (BOX likely)
    reason = BettingStrategy.generate_reason(
        'umaren', ['1', '2'], 
        prob=0.1, ev=0.0, odds=0.0, features_list=[{'avg_last_3f': 34.0}]
    )
    print(f"[Umaren 0.0 with Feature] {reason}")
    assert "0.00倍" not in reason
    assert "0.0倍" not in reason
    assert "上位拮抗" in reason

if __name__ == "__main__":
    try:
        test_generate_reason()
        print("Test Passed: '0.0x' messages successfully removed.")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
