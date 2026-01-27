import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""エスケープシーケンスの修正スクリプト"""

filepath = r'modules\netkeiba_automator.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 誤ったエスケープシーケンスを修正
content = content.replace(r'f\"', 'f"')
content = content.replace(r'\"', '"')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed escaping issues in", filepath)
