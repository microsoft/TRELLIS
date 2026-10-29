# Run TRELLIS image-to-3D. Optional args: one or more image paths. Log: logs\test.log
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Images = @())
. (Join-Path $PSScriptRoot 'common.ps1')
Start-Log 'test.log'
$ok = $false
try {
    $CondaRoot = Find-Conda
    if (-not $CondaRoot) { throw 'conda not found' }
    $EnvPath = Get-EnvPath (Join-Path $CondaRoot 'Scripts\conda.exe')
    if (-not $EnvPath) { throw "conda env '$EnvName' not found - run 1_install.bat first" }
    Use-Env $EnvPath
    # Inference needs no compiler or CUDA Toolkit: the CUDA extensions were compiled at install time
    $env:ATTN_BACKEND = 'xformers'
    $env:SPCONV_ALGO = 'native'
    $a = @((Join-Path $Here 'run_test.py'))
    foreach ($img in $Images) { if ($img) { $a += $img } }
    Exec $Py $a
    $ok = $true
    Log 'TEST OK' 'Green'
}
catch { Log "TEST FAILED: $_" 'Red' }
finally { Log "finished $(Get-Date -Format o)"; Stop-Log }
if ($ok) { exit 0 } else { exit 1 }
