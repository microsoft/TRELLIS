# TRELLIS — Gradio demo, conda-free, CUDA 12.1 (RTX 4090 / Ada sm_89).
# Build:  docker build -t trellis .
# Run:    see docker-compose.yml or the `docker run` line in README.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# 4090 = Ada = compute 8.9. Pin arch so extension builds are fast and small.
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV ATTN_BACKEND=xformers
ENV SPCONV_ALGO=native
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip wheel setuptools

WORKDIR /app

# --- PyTorch (cu121, matches CUDA 12.1 toolkit) ---
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- basic deps ---
RUN pip install "numpy<2" \
        pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
        rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph \
        "transformers==4.46.3" "huggingface_hub==0.25.2" \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- attention + sparse backends (prebuilt wheels) ---
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install spconv-cu120

# --- rendering extensions (compiled with nvcc) ---
# --no-build-isolation: these setup.py files import torch at build time,
# which build isolation would otherwise hide. Provide setuptools too.
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
RUN pip install --no-build-isolation git+https://github.com/JeffreyXiang/diffoctreerast.git
RUN pip install --no-build-isolation "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization"

# --- Gradio demo ---
# Pin the whole web stack to gradio-4.44.1-era versions. Newer fastapi/starlette/
# pydantic (pulled transitively) break gradio 4.44 (schema parser + TemplateResponse
# API changes), so pin them explicitly to avoid runtime crashes.
RUN pip install gradio==4.44.1 gradio_litmodel3d==0.0.1 \
        "fastapi==0.112.2" "starlette==0.38.6" "pydantic==2.9.2" "pydantic-core==2.23.4"

# --- kaolin (required by the flexicubes submodule for mesh extraction) ---
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Patch a known gradio 4.44.1 bug: gradio_client schema parser crashes on
# boolean `additionalProperties` (raises APIInfoParseError "Cannot parse schema True"),
# which breaks the launch() localhost health-check. Make the parser tolerant.
RUN python - <<'PYEOF'
import pathlib
F = pathlib.Path("/usr/local/lib/python3.10/dist-packages/gradio_client/utils.py")
s = F.read_text()
s = s.replace(
    '    if "const" in schema:',
    '    if not isinstance(schema, dict):\n        return "Any"\n    if "const" in schema:')
s = s.replace(
    'raise APIInfoParseError(f"Cannot parse schema {schema}")',
    'return "Any"')
F.write_text(s)
assert 'Cannot parse schema {schema}' not in s, "gradio_client patch FAILED"
print("PATCHED gradio_client OK")
PYEOF

# Copy the repo last so code edits don't bust the dependency cache.
# IMPORTANT: run `git submodule update --init --recursive` on the host BEFORE build
# so trellis/representations/mesh/flexicubes is populated and gets copied in.
COPY . /app

# HF model cache persists via the mounted volume (see docker run / compose).
ENV HF_HOME=/app/.hf_cache

EXPOSE 7860
# Gradio must listen on 0.0.0.0 to be reachable outside the container.
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "app.py"]
