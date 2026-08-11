<# 启动本地 pytest-watch；优先复用已经激活的虚拟环境。 #>

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if (-not $env:VIRTUAL_ENV) {
    $workspaceRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
    $activateCandidates = @(
        (Join-Path $projectRoot ".venv\Scripts\Activate.ps1"),
        (Join-Path $projectRoot "venv\Scripts\Activate.ps1"),
        (Join-Path $workspaceRoot "venv\Scripts\Activate.ps1")
    )
    $activateScript = $activateCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1

    if ($activateScript) {
        Write-Host "激活虚拟环境：$activateScript"
        . $activateScript
    }
}

if (-not (Get-Command ptw -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 pytest-watch。请先运行：python -m pip install -r requirements.txt"
}

Write-Host "监听项目中的 Python 文件变化，并自动执行 pytest tests/。"
Write-Host "按 Ctrl+C 停止。"
& ptw --runner "python -m pytest tests/"
