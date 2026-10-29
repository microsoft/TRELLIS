# Shared helpers for the TRELLIS Windows scripts (Windows PowerShell 5.1 compatible, ASCII only)
$ErrorActionPreference = 'Continue'
$Here    = $PSScriptRoot
$Repo    = Split-Path -Parent $PSScriptRoot
$LogDir  = Join-Path $Here 'logs'
$CudaVer = 'v12.6'
$EnvName = 'trellis'
New-Item -ItemType Directory -Force $LogDir | Out-Null

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_NO_INPUT = '1'

function Start-Log([string]$name) {
    $script:LogPath = Join-Path $LogDir $name
    $script:W = New-Object System.IO.StreamWriter($script:LogPath, $false, (New-Object System.Text.UTF8Encoding($false)))
    $script:W.AutoFlush = $true
    Log "started $(Get-Date -Format o)"
}
function Stop-Log { if ($script:W) { $script:W.Close(); $script:W = $null } }
function Log([string]$msg, [string]$color = 'Gray') {
    Write-Host $msg -ForegroundColor $color
    if ($script:W) { $script:W.WriteLine($msg) }
}
function Step([string]$title) { Log ''; Log "===== $title =====" 'Cyan' }

# Run a native command, streaming stdout+stderr to console and log. Sets $script:LastCode.
function Exec {
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$ArgList = @(), [switch]$AllowFail)
    Log ('> ' + $Exe + ' ' + ($ArgList -join ' ')) 'DarkGray'
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe @ArgList 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $line = $_.Exception.Message } else { $line = [string]$_ }
        Write-Host $line
        if ($script:W) { $script:W.WriteLine($line) }
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    $script:LastCode = $code
    if (($code -ne 0) -and (-not $AllowFail)) { throw "Command failed with exit code ${code}: $Exe $($ArgList -join ' ')" }
}

function Find-Conda {
    $roots = New-Object System.Collections.Generic.List[string]
    if ($env:CONDA_EXE) { $roots.Add((Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent)) }
    foreach ($p in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\Miniconda3", "$env:USERPROFILE\anaconda3",
                     "$env:LOCALAPPDATA\miniconda3", "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\Continuum\miniconda3",
                     "C:\ProgramData\miniconda3", "C:\ProgramData\Miniconda3", "C:\ProgramData\anaconda3",
                     "C:\miniconda3", "C:\Miniconda3", "D:\miniconda3", "D:\Miniconda3", "C:\tools\miniconda3")) { $roots.Add($p) }
    $sh = New-Object -ComObject WScript.Shell
    foreach ($sm in @("$env:APPDATA\Microsoft\Windows\Start Menu\Programs", "$env:ProgramData\Microsoft\Windows\Start Menu\Programs")) {
        if (Test-Path $sm) {
            Get-ChildItem $sm -Recurse -Filter *.lnk -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'conda' } | ForEach-Object {
                $lnk = $sh.CreateShortcut($_.FullName)
                $text = $lnk.Arguments + ' ' + $lnk.TargetPath
                foreach ($m in [regex]::Matches($text, '([A-Za-z]:\\[^''"]+?)\\(shell\\condabin|Scripts\\activate|condabin)')) { $roots.Add($m.Groups[1].Value) }
            }
        }
    }
    $envTxt = "$env:USERPROFILE\.conda\environments.txt"
    if (Test-Path $envTxt) {
        foreach ($l in (Get-Content $envTxt)) {
            $t = $l.Trim()
            if ($t) { $roots.Add($t); $roots.Add((Split-Path (Split-Path $t -Parent) -Parent)) }
        }
    }
    foreach ($r in $roots) {
        if ($r -and (Test-Path (Join-Path $r 'Scripts\conda.exe'))) { return (Resolve-Path $r).Path }
    }
    return $null
}

function Get-EnvPath([string]$conda) {
    $raw = & $conda env list --json 2>$null | Out-String
    try { $envs = ($raw | ConvertFrom-Json).envs } catch { $envs = @() }
    foreach ($e in $envs) { if ((Split-Path $e -Leaf) -ieq $EnvName) { return $e } }
    return $null
}

function Use-Env([string]$envPath) {
    $script:EnvPath = $envPath
    $script:Py = Join-Path $envPath 'python.exe'
    $prefix = @($envPath, "$envPath\Library\mingw-w64\bin", "$envPath\Library\usr\bin", "$envPath\Library\bin", "$envPath\Scripts", "$envPath\bin") -join ';'
    $env:PATH = $prefix + ';' + $env:PATH
    $env:CONDA_PREFIX = $envPath
    Log "python: $script:Py"
}

# Load the VS 2022 x64 compiler environment (CUDA 12.6 does not accept the VS 2026 compiler)
function Import-VsEnv {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found - is Visual Studio installed?' }
    $vs = & $vswhere -products * -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    if (-not $vs) { throw 'Visual Studio 2022 with the C++ x64 build tools was not found.' }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    Log "vcvars: $vcvars"
    $lines = cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($l in $lines) {
        if ($l -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { throw 'cl.exe is not on PATH after running vcvars64.bat' }
    Log "cl.exe: $($cl.Source)  (VCToolsVersion=$env:VCToolsVersion)"
}

# Compute capabilities of the installed NVIDIA GPUs, e.g. '8.6+PTX' or '7.5;8.9+PTX'. $null if nvidia-smi can't tell.
function Get-GpuArchList {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    $caps = @(& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
              ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+\.\d+$' } |
              Select-Object -Unique | Sort-Object { [version]$_ })
    if ($caps.Count -eq 0) { return $null }
    # CUDA 12.6 nvcc stops at sm_90; Blackwell (sm_100 / sm_120) needs CUDA 12.8+ and matching torch wheels
    $newest = $caps[-1]
    if ([version]$newest -gt [version]'9.0') { throw "GPU compute capability $newest is not supported by CUDA $CudaVer (max 9.0)" }
    $caps[-1] = "$newest+PTX"
    return ($caps -join ';')
}

function Set-CudaEnv {
    $cuda = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\$CudaVer"
    if (-not (Test-Path (Join-Path $cuda 'bin\nvcc.exe'))) { throw "CUDA Toolkit $CudaVer not found at $cuda" }
    $env:CUDA_HOME = $cuda
    $env:CUDA_PATH = $cuda
    $env:PATH = (Join-Path $cuda 'bin') + ';' + (Join-Path $cuda 'libnvvp') + ';' + $env:PATH
    # Build the CUDA extensions for this machine's GPUs unless the caller already chose an arch list
    if ($env:TORCH_CUDA_ARCH_LIST) {
        $archSrc = 'preset by caller/environment'
    } else {
        $arch = Get-GpuArchList
        if ($arch) { $env:TORCH_CUDA_ARCH_LIST = $arch; $archSrc = 'detected with nvidia-smi' }
        else {
            $archSrc = 'unset'
            Log 'WARNING: nvidia-smi did not report a compute capability; PyTorch will detect the GPU at build time' 'Yellow'
        }
    }
    $env:DISTUTILS_USE_SDK = '1'
    if (-not $env:MAX_JOBS) { $env:MAX_JOBS = '8' }
    Log "CUDA_HOME=$cuda  TORCH_CUDA_ARCH_LIST=$env:TORCH_CUDA_ARCH_LIST ($archSrc)"
}
