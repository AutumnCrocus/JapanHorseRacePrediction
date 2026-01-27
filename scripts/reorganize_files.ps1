# ファイル整理スクリプト
$ErrorActionPreference = "Stop"
$root = "C:\Users\t4kic\Documents\ネット競馬"
$scriptsDir = Join-Path $root "scripts"

# 1. フォルダ作成
$folders = @("core", "simulation", "analysis", "debug", "legacy")
foreach ($f in $folders) {
    $path = Join-Path $scriptsDir $f
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "Created $path"
    }
}

# 2. ファイル定義 (移動ルール)
$moveRules = @{
    "core" = @(
        "predict_tomorrow.py", "train_production.py", "train_model_improved.py", "evaluate_prediction.py"
    )
    "simulation" = @(
        "simulate_*.py", "run_rolling_simulation.py", "generate_simulation_data.py"
    )
    "analysis" = @(
        "analyze_*.py", "summarize_*.py", "extract_strategy_b.py", "inspect_*.py"
    )
    "debug" = @(
        "debug_*.py", "check_*.py", "diagnose_*.py", "test_*.py", "verify_*.py"
    )
    "legacy" = @(
        "train_period.py", "recover_bet_details.py"
    )
}

# ルートからの移動分
$rootToDebug = @("check_cols.py", "create_prediction_csv.py", "fix_escaping.py")
foreach ($file in $rootToDebug) {
    $src = Join-Path $root $file
    $dest = Join-Path $scriptsDir "debug"
    if (Test-Path $src) {
        Move-Item $src $dest -Force
        Write-Host "Moved $file to scripts/debug/"
        
        # モジュールパス修正(ルートからscripts/debugへ = 2階層下がるが、sys.path的には...)
        # 元: なし(カレント想定) or sys.path.append(...)
        # 移動後: scripts/debug/xxx.py -> rootは 3階層上
        
        $content = Get-Content (Join-Path $dest $file) -Raw
        if ($content -notmatch "sys.path") {
            $header = "import sys`nimport os`nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))`n"
            $newContent = $header + $content
            Set-Content (Join-Path $dest $file) -Value $newContent -Encoding UTF8
        }
    }
}

# scripts内の移動
cd $scriptsDir
$allPyFiles = Get-ChildItem -Filter *.py
foreach ($file in $allPyFiles) {
    $fname = $file.Name
    $targetDir = ""
    
    foreach ($key in $moveRules.Keys) {
        foreach ($pattern in $moveRules[$key]) {
            if ($fname -like $pattern) {
                $targetDir = $key
                break
            }
        }
        if ($targetDir) { break }
    }
    
    if ($targetDir) {
        $destPath = Join-Path $scriptsDir $targetDir
        $destFile = Join-Path $destPath $fname
        Move-Item $file.FullName $destFile -Force
        Write-Host "Moved $fname to $targetDir/"
        
        # sys.path修正
        # 元: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) (2回dirname)
        # 先: scripts/subdir/file.py -> rootは 3回dirnameが必要
        
        $content = Get-Content $destFile -Raw
        # 単純な置換：dirname(dirname( -> dirname(dirname(dirname(
        # ただし既に3回ある場合などを考慮して、特定のパターン「dirname(os.path.abspath(__file__))」 の前にもう一個dirnameをつける
        
        if ($content -match "os.path.dirname\(os.path.dirname\(os.path.abspath\(__file__\)\)\)") {
            $newContent = $content -replace "os.path.dirname\(os.path.dirname\(os.path.abspath\(__file__\)\)\)", "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
            Set-Content $destFile -Value $newContent -Encoding UTF8
            Write-Host "  Updated sys.path in $fname"
        }
    }
}

# 3. README生成
$readmePath = Join-Path $scriptsDir "README_FILE_STRUCTURE.md"
$readmeContent = @"
# Scripts Directory Structure

This directory contains various scripts for the JapanHorseRacePrediction project.

## 📂 core
Production-ready scripts essential for the application lifecycle.
- Prediction, Training, Evaluation

## 📂 simulation
Scripts for simulating betting strategies and calculating recovery rates.
- `simulate_graded_30patterns.py`: **Main script for Graded Race Strategy**

## 📂 analysis
Tools for analyzing data distributions, prize money, and filtering logic.

## 📂 debug
Scripts for debugging, quick verification, and temporary checks.
- Contains one-off scripts and diagnostic tools.

## 📂 legacy
Deprecated scripts kept for reference.

"@
Set-Content $readmePath -Value $readmeContent -Encoding UTF8
Write-Host "Created README_FILE_STRUCTURE.md"
