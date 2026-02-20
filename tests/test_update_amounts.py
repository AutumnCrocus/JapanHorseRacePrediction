import sys
import os
import pytest
import re

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reports.update_amounts import update_markdown_amounts, process_block

def test_process_block_scaling():
    """金額が5000円から遠い場合はスケールされることの確認"""
    sample_block = """### モデル: lgbm/dummy
- 馬連 1-2: 1500円 (15点)
- ワイド 1-3: 1500円 (15点)
- 単勝 1: 1500円 (15点)
"""
    # Total = 4500円. Target = 5000円
    # 5000 * 1500 / 4500 = 1666.6 -> 1700円
    expected_block = """### モデル: lgbm/dummy
- 馬連 1-2: 1700円 (15点)
- ワイド 1-3: 1700円 (15点)
- 単勝 1: 1700円 (15点)
"""
    
    match = re.search(r'(### モデル: .+?\n)(.*?)(?=(?:### |\Z))', sample_block, flags=re.DOTALL)
    assert match is not None
    
    result = process_block(match)
    assert result == expected_block

def test_update_markdown_amounts_no_change_if_close():
    """金額がすでに5000円に近い場合はそのままになることの確認"""
    sample_text = """### モデル: lgbm/box4_sanrenpuku
- 3連複 1,2,3,4ボックス: 5000円 (4点)
### 
"""
    # Total = 5000 -> No change expected
    expected_text = sample_text
    result = update_markdown_amounts(sample_text)
    assert result == expected_text

def test_update_markdown_amounts_full_text():
    """複数ブロックが含まれるテキスト全体に対する置換処理の確認"""
    sample_text = """### モデル: A
- 枠連 1-2: 3000円 (30点)
### モデル: B
- 単勝 5: 1000円 (1点)
"""
    # Block A: Total 3000 -> 5000
    # Block B: Total 1000 -> 5000
    expected_text = """### モデル: A
- 枠連 1-2: 5000円 (30点)
### モデル: B
- 単勝 5: 5000円 (1点)
"""
    result = update_markdown_amounts(sample_text)
    assert result == expected_text
