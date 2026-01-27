# ファイル整理スクリプト (Python版)
import os
import shutil
import glob
import re

def reorganize():
    # ルートディレクトリ取得 (scriptsフォルダ内にある前提)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(scripts_dir)
    
    print(f"Scripts Dir: {scripts_dir}")
    print(f"Root Dir: {root_dir}")
    
    # 1. フォルダ作成
    folders = ["core", "simulation", "analysis", "debug", "legacy"]
    for f in folders:
        path = os.path.join(scripts_dir, f)
        os.makedirs(path, exist_ok=True)
        print(f"Ensure dir: {path}")

    # 2. 移動ルール
    move_rules = {
        "core": [
            "predict_tomorrow.py", "train_production.py", "train_model_improved.py", "evaluate_prediction.py"
        ],
        "simulation": [
            "simulate_*.py", "run_rolling_simulation.py", "generate_simulation_data.py"
        ],
        "analysis": [
            "analyze_*.py", "summarize_*.py", "extract_strategy_b.py", "inspect_*.py"
        ],
        "debug": [
            "debug_*.py", "check_*.py", "diagnose_*.py", "test_*.py", "verify_*.py"
        ],
        "legacy": [
            "train_period.py", "recover_bet_details.py"
        ]
    }
    
    # helper: update sys.path
    def update_sys_path(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # パターン: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # これを dirname(dirname(dirname(...))) に置換
            # 正規表現で dirname のネスト数を増やす
            
            # 単純な search & replace
            old_str = "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
            new_str = "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
            
            if old_str in content and new_str not in content:
                new_content = content.replace(old_str, new_str)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  Updated import path in {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  Error updating {os.path.basename(file_path)}: {e}")

    # scripts内の移動
    for py_file in glob.glob(os.path.join(scripts_dir, "*.py")):
        fname = os.path.basename(py_file)
        if fname == os.path.basename(__file__): continue # 自分自身は移動しない
        
        target_sub = None
        for key, patterns in move_rules.items():
            for pat in patterns:
                # glob pattern matching
                # fnmatch style
                import fnmatch
                if fnmatch.fnmatch(fname, pat):
                    target_sub = key
                    break
            if target_sub: break
        
        if target_sub:
            dest = os.path.join(scripts_dir, target_sub, fname)
            try:
                shutil.move(py_file, dest)
                print(f"Moved {fname} -> {target_sub}/")
                update_sys_path(dest)
            except Exception as e:
                print(f"Failed to move {fname}: {e}")

    # ルートにあるファイルの移動 (check_cols.py など)
    root_moves = ["check_cols.py", "create_prediction_csv.py", "fix_escaping.py"]
    for fname in root_moves:
        src = os.path.join(root_dir, fname)
        if os.path.exists(src):
            dest = os.path.join(scripts_dir, "debug", fname)
            try:
                shutil.move(src, dest)
                print(f"Moved {fname} -> scripts/debug/")
                
                # ルートから scripts/debug に移動する場合、importパスを追加する必要がある
                # 元ファイルには sys.path の設定がない場合が多い
                with open(dest, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                header = "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\n"
                
                if "sys.path" not in content:
                    with open(dest, 'w', encoding='utf-8') as f:
                        f.write(header + content)
                        
            except Exception as e:
                print(f"Failed to move {fname}: {e}")

if __name__ == "__main__":
    reorganize()
