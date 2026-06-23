#!/usr/bin/env bash
# Conda-free TRELLIS install for the Gradio demo (app.py).
# Target: WSL2 Ubuntu 22.04, NVIDIA GPU, CUDA 11.8 toolkit, Python 3.10.
# Replaces setup.sh --basic --xformers --spconv --nvdiffrast --diffoctreerast --mipgaussian --demo
# (kaolin + flash-attn intentionally omitted; see plan.)
set -e

# --- 0. prerequisites (run once, outside this script) ---------------------
# sudo apt update
# sudo apt install -y build-essential git python3.10 python3.10-venv python3-pip ffmpeg libgl1 libglib2.0-0
# install CUDA toolkit 11.8 via NVIDIA's WSL-Ubuntu repo (cuda-toolkit-11-8)
# verify: nvidia-smi && nvcc --version  (expect 11.8)

# correct nvcc on PATH for source builds
export PATH=/usr/local/cuda-11.8/bin:$PATH

# --- 1. submodules --------------------------------------------------------
git submodule update --init --recursive   # flexicubes (mesh extraction)

# --- 2. venv --------------------------------------------------------------
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# --- 3. PyTorch (cu118) ---------------------------------------------------
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# --- 4. basic deps --------------------------------------------------------
pip install "numpy<2" \
            pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
            rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- 5. attention + sparse backends (prebuilt wheels) ---------------------
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu118

# --- 6. rendering extensions (compiled with nvcc, slowest step) -----------
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/JeffreyXiang/diffoctreerast.git
pip install "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization"

# --- 7. Gradio demo -------------------------------------------------------
pip install gradio==4.44.1 gradio_litmodel3d==0.0.1

echo
echo "Done. To run the demo:"
echo "  source .venv/bin/activate"
echo "  export ATTN_BACKEND=xformers SPCONV_ALGO=native PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python app.py"
