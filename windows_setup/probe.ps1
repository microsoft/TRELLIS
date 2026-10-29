# Probe the Windows toolchain needed for TRELLIS. Writes logs\probe.log
$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot 'logs\probe.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Set-Content -Path $log -Value "probe started $(Get-Date -Format o)" -Encoding UTF8
function Run($title, [scriptblock]$sb) {
    $hdr = "`r`n===== $title ====="
    Write-Host $hdr -ForegroundColor Cyan
    Add-Content -Path $log -Value $hdr -Encoding UTF8
    try { $out = & $sb 2>&1 | Out-String -Width 400 } catch { $out = "ERROR: $_" }
    Write-Host $out
    Add-Content -Path $log -Value $out -Encoding UTF8
}
Run 'OS' { (Get-CimInstance Win32_OperatingSystem).Caption; [Environment]::OSVersion.VersionString; "RAM GB: " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1) }
Run 'GPU (nvidia-smi)' { nvidia-smi }
Run 'GPU query' { nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv }
Run 'CUDA env vars' { Get-ChildItem env: | Where-Object { $_.Name -like 'CUDA*' } | Format-Table Name,Value -AutoSize -Wrap }
Run 'CUDA toolkits installed' { Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName }
Run 'nvcc' { where.exe nvcc; nvcc --version }
Run 'git' { where.exe git; git --version }
Run 'conda locations' {
    "CONDA_EXE=$env:CONDA_EXE"
    where.exe conda
    foreach ($p in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniconda3", "C:\ProgramData\anaconda3", "C:\miniconda3")) { if (Test-Path "$p\Scripts\conda.exe") { "FOUND $p" } }
}
Run 'conda envs' {
    $c = $env:CONDA_EXE
    if (-not $c) { foreach ($p in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniconda3", "C:\ProgramData\anaconda3")) { if (Test-Path "$p\Scripts\conda.exe") { $c = "$p\Scripts\conda.exe"; break } } }
    if ($c) { & $c --version; & $c env list } else { 'conda.exe not found' }
}
Run 'Visual Studio / MSVC' {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        & $vswhere -all -products * -format text -property displayName
        & $vswhere -all -products * -format text -property installationPath
        & $vswhere -all -products * -format text -property installationVersion
        'With VC x64 tools:'
        & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        foreach ($ip in (& $vswhere -all -products * -property installationPath)) { Get-ChildItem "$ip\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName }
    } else { 'vswhere not found' }
}
Run 'python on PATH' { where.exe python; where.exe py }
Run 'Disk' { Get-PSDrive -PSProvider FileSystem | Format-Table Name,@{n='UsedGB';e={[math]::Round($_.Used/1GB,1)}},@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} -AutoSize }
Run 'HF cache' { "HF_HOME=$env:HF_HOME"; Test-Path "$env:USERPROFILE\.cache\huggingface" }
Run 'LongPathsEnabled' { (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled }
Add-Content -Path $log -Value "`r`nPROBE DONE $(Get-Date -Format o)" -Encoding UTF8
Write-Host "`nDone. Log written to $log" -ForegroundColor Green
