# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,gsplat,demo -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
LINEAR_ASSIGNMENT=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
GSPLAT=false
DEMO=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --gsplat) GSPLAT=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --gsplat                Install gsplat"
    echo "  --demo                  Install all dependencies for demo"
    return
fi

if [ "$NEW_ENV" = true ] ; then
    conda create -n trellis python=3.10
    conda activate trellis
    conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
fi

# Get system information
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    hip)
        HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
        HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
        HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
        # Install pytorch 2.4.1 for hip
        if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ] ; then
        echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
            pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1 --user
            mkdir -p /tmp/extensions
            sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
            cd /tmp/extensions/amd_smi
            sudo chmod -R 777 .
            pip install .
            cd $WORKDIR
            PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        fi
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
        ;;
    *)
        ;;
esac

if [ "$BASIC" = true ] ; then
    pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

if [ "$XFORMERS" = true ] ; then
    # install xformers
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ "$CUDA_VERSION" = "11.8" ] ; then
            case $PYTORCH_VERSION in
                2.0.1) pip install https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl ;;
                2.1.0) pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.1) pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.2) pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.0) pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.1) pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.2) pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.3.0) pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.0) pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ] ; then
            case $PYTORCH_VERSION in
                2.1.0) pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.1) pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.2) pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.0) pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.1) pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.2) pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.3.0) pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.0) pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ] ; then
            case $PYTORCH_VERSION in
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_MAJOR_VERSION"
        fi
    elif [ "$PLATFORM" = "hip" ] ; then
        case $PYTORCH_VERSION in
            2.4.1\+rocm6.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1 ;;
            *) echo "[XFORMERS] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install flash-attn
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.6.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$KAOLIN" = true ] ; then
    # install kaolin
    if [ "$PLATFORM" = "cuda" ] ; then
        case $PYTORCH_VERSION in
            2.0.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html;;
            2.1.0) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html;;
            2.1.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html;;
            2.2.0) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html;;
            2.2.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html;;
            2.2.2) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html;;
            2.4.0) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html;;
            *) echo "[KAOLIN] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DIFFOCTREERAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
        pip install /tmp/extensions/diffoctreerast
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$MIPGAUSSIAN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
        pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$VOX2SEQ" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        cp -r extensions/vox2seq /tmp/extensions/vox2seq
        pip install /tmp/extensions/vox2seq
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$SPCONV" = true ] ; then
    # install spconv
    if [ "$PLATFORM" = "cuda" ] ; then
        case $CUDA_MAJOR_VERSION in
            11) pip install spconv-cu118 ;;
            12) pip install spconv-cu120 ;;
            *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION" ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DEMO" = true ] ; then
    pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
fi

if [ "$GSPLAT" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install git+https://github.com/nerfstudio-project/gsplat
    else
        echo "[GSPLAT] Unsupported platform: $PLATFORM"
    fi
fi