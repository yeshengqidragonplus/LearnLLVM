# Windows 便捷入口；其他平台直接运行 setup_env.py。
param(
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Backend = 'auto',
    [string]$Cuda = 'cu126',
    [string]$Python = 'py',
    [string]$Venv = '.venv',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$setupArgs = @((Join-Path $PSScriptRoot 'setup_env.py'), '--backend', $Backend, '--cuda', $Cuda, '--venv', $Venv)
if ($DryRun) { $setupArgs += '--dry-run' }
if ($Python -eq 'py') { & py -3.11 @setupArgs } else { & $Python @setupArgs }
if ($LASTEXITCODE -ne 0) { throw '环境安装或验证失败' }
