# 不要ファイル一括削除スクリプト

$deleteList = @(
    # 俺プロ関連 (scripts内)
    "scripts/dump_orepro.py",
    "scripts/dump_orepro_ipat.py",
    "scripts/dump_orepro_manual.py",
    "scripts/orepro_auto_vote.py",
    
    # HTMLダンプ
    "orepro_bet.html",
    "orepro_ipat.html",
    "orepro_target_page.html",
    "ipat_page_after_fuku_tab.html",
    "ipat_page_after_tan_tab.html",
    "ipat_page_before_tab.html",
    
    # 解析・一時ファイル
    "modes_analysis_result.txt",
    "scripts/save_date_analysis.py",
    "scripts/run_automator_directly.py",
    "scripts/run_automator_multi.py",
    "scripts/trigger_ipat_api.py",
    "scripts/verify_complex_bets.py"
)

# ワイルドカード削除
$patterns = @(
    "scripts/analyze_*.py",
    "scripts/inspect_*.py",
    "scripts/check_*.py",
    "scripts/debug_*.py",
    "scripts/find_*.py",
    "scripts/test_*.py"
)

Write-Host "Deleting specific files..."
foreach ($file in $deleteList) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "Deleted: $file"
    }
}

Write-Host "Deleting patterns..."
foreach ($pattern in $patterns) {
    $files = Get-ChildItem $pattern -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        Remove-Item $f.FullName -Force
        Write-Host "Deleted: $($f.Name)"
    }
}

# ディレクトリ削除
if (Test-Path "scripts/debug_screenshots") {
    Remove-Item "scripts/debug_screenshots" -Recurse -Force
    Write-Host "Deleted: scripts/debug_screenshots"
}
if (Test-Path "scripts/test_screenshots") {
    Remove-Item "scripts/test_screenshots" -Recurse -Force
    Write-Host "Deleted: scripts/test_screenshots"
}

# 統合処理: generate_betting_list.py -> scripts/predict_tomorrow.pyに統合済みと仮定して削除
# (今回はまだ統合コードを書いていないが、統合は後回しにして削除だけはしないでおく、または削除リストに追加済みなら削除)
# ユーザー要望で「統合」となっているので、generate_betting_list.pyは残しておくか、
# あるいはこのスクリプト内で消すか。
# 今回は消さずに残す。

Write-Host "Cleanup completed."
