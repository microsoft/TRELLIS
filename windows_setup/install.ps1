# TRELLIS Windows installer. Safe to re-run: finished steps are skipped. Log: logs\install.log
. (Join-Path $PSScriptRoot 'common.ps1')
Start-Log 'install.log'
$ok = $false
try {
    Step 'Locate conda'
    $CondaRoot = Find-Conda
    if (-not $CondaRoot) { throw 'Could not find Miniconda/Anaconda (Scripts\conda.exe).' }
    $Conda = Join-Path $CondaRoot 'Scripts\conda.exe'
    Log "conda: $Conda"
    Exec $Conda @('--version')

    Step 'Build environment: Visual Studio 2022 + CUDA 12.6'
    Import-VsEnv
    Set-CudaEnv
    Exec 'nvcc' @('--version')

    Step 'FlexiCubes git submodule'
    if (Test-Path (Join-Path $Repo 'trellis\representations\mesh\flexicubes\flexicubes.py')) { Log 'already present' }
    else { Exec 'git' @('-c', 'safe.directory=*', '-C', $Repo, 'submodule', 'update', '--init', '--recursive') }

    Step "Conda env '$EnvName' (Python 3.10)"
    $EnvPath = Get-EnvPath $Conda
    if (-not $EnvPath) {
        Exec $Conda @('create', '-y', '-n', $EnvName, 'python=3.10', 'pip', '-c', 'conda-forge', '--override-channels')
        $EnvPath = Get-EnvPath $Conda
    } else { Log "reusing existing env at $EnvPath" }
    if (-not $EnvPath) { throw "conda env '$EnvName' was not created" }
    Use-Env $EnvPath
    Exec $Py @('-c', 'import sys; print(sys.version); assert sys.version_info[:2] in [(3,10),(3,11)], sys.version')

    $Pip  = @('-m', 'pip', 'install', '--progress-bar', 'off')
    $Cons = Join-Path $Here 'constraints.txt'
    Exec $Py ($Pip + @('--upgrade', 'pip', 'wheel', 'setuptools<80'))

    Step 'PyTorch 2.6.0 (CUDA 12.6)'
    Exec $Py @('-c', "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.6.0') and torch.version.cuda == '12.6' and torch.cuda.is_available() else 1)") -AllowFail
    if ($LastCode -ne 0) { Exec $Py ($Pip + @('torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu126')) }
    Exec $Py @('-c', "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0))")

    Step 'xformers'
    Exec $Py @('-c', 'import xformers.ops') -AllowFail
    if ($LastCode -ne 0) {
        Exec $Py ($Pip + @('xformers==0.0.29.post3', '--no-deps', '--only-binary', ':all:', '--index-url', 'https://download.pytorch.org/whl/cu126')) -AllowFail
        if ($LastCode -ne 0) { Exec $Py ($Pip + @('xformers==0.0.29.post3', '--no-deps', '--only-binary', ':all:')) }
    }

    Step 'Python dependencies'
    Exec $Py ($Pip + @('-c', $Cons, 'pillow', 'imageio', 'imageio-ffmpeg', 'tqdm', 'easydict', 'opencv-python-headless', 'scipy',
                       'ninja', 'rembg', 'onnxruntime', 'trimesh', 'open3d', 'xatlas', 'pyvista', 'pymeshfix', 'igraph',
                       'transformers', 'huggingface_hub', 'safetensors'))
    Exec $Py @('-c', 'import utils3d') -AllowFail
    if ($LastCode -ne 0) { Exec $Py ($Pip + @('-c', $Cons, 'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8')) }

    Step 'pymeshfix API check (TRELLIS needs pymeshfix._meshfix.PyTMesh)'
    Exec $Py @('-c', 'from pymeshfix import _meshfix; _meshfix.PyTMesh(); print(1)') -AllowFail
    if ($LastCode -ne 0) {
        Exec $Py ($Pip + @('-c', $Cons, 'pymeshfix==0.16.3')) -AllowFail
        if ($LastCode -ne 0) { Exec $Py ($Pip + @('-c', $Cons, 'pymeshfix<0.17')) }
    }

    Step 'spconv'
    Exec $Py @('-c', 'import torch, spconv.pytorch') -AllowFail
    if ($LastCode -ne 0) { Exec $Py ($Pip + @('-c', $Cons, 'spconv-cu120')) }

    $Ext = Join-Path $Here 'ext'
    New-Item -ItemType Directory -Force $Ext | Out-Null

    Step 'nvdiffrast (CUDA extension build)'
    Exec $Py @('-c', 'import nvdiffrast.torch') -AllowFail
    if ($LastCode -ne 0) {
        $nvd = Join-Path $Ext 'nvdiffrast'
        if (-not (Test-Path (Join-Path $nvd '.git'))) { Exec 'git' @('clone', '--depth', '1', 'https://github.com/NVlabs/nvdiffrast.git', $nvd) }
        Exec $Py ($Pip + @('-c', $Cons, '--no-build-isolation', '-v', $nvd))
    }

    Step 'diff-gaussian-rasterization from mip-splatting (CUDA extension build)'
    Exec $Py @('-c', 'import diff_gaussian_rasterization') -AllowFail
    if ($LastCode -ne 0) {
        $mip = Join-Path $Ext 'mip-splatting'
        if (-not (Test-Path (Join-Path $mip '.git'))) { Exec 'git' @('clone', '--depth', '1', 'https://github.com/autonomousvision/mip-splatting.git', $mip) }
        $dgr = Join-Path $mip 'submodules\diff-gaussian-rasterization'
        if (-not (Test-Path (Join-Path $dgr 'third_party\glm\glm\glm.hpp'))) {
            Exec 'git' @('clone', '--depth', '1', 'https://github.com/g-truc/glm.git', (Join-Path $dgr 'third_party\glm'))
        }
        Exec $Py ($Pip + @('-c', $Cons, '--no-build-isolation', '-v', $dgr))
    }

    Step 'Verify environment'
    $env:ATTN_BACKEND = 'xformers'
    $env:SPCONV_ALGO = 'native'
    Exec $Py @((Join-Path $Here 'check_env.py'))
    $ok = $true
    Log ''
    Log 'INSTALL OK' 'Green'
}
catch {
    Log ''
    Log "INSTALL FAILED: $_" 'Red'
}
finally {
    Log "finished $(Get-Date -Format o)"
    Stop-Log
}
if ($ok) { exit 0 } else { exit 1 }
