# TRELLIS on Windows

Scripts that install and run the TRELLIS **image-to-3D** pipeline natively on Windows (no WSL),
without kaolin, flash-attn or diffoctreerast. Tested on Windows 11 with an RTX 3080 Laptop GPU (16 GB).

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11, 64-bit |
| GPU | NVIDIA, compute capability 7.5 - 9.0 (RTX 20 / 30 / 40 series, A-series, Hopper). 16 GB VRAM recommended; RTX 50-series (Blackwell) needs CUDA 12.8+ and is not supported by this stack |
| CUDA Toolkit | **12.6**, installed in the default location (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`) |
| Compiler | **Visual Studio 2022** (Community is fine) with the *Desktop development with C++* workload. VS 2026's compiler is too new for CUDA 12.6; the scripts pick VS 2022 even if both are installed |
| Python | Miniconda or Anaconda (found automatically; does not need to be on `PATH`) |
| Other | Git, ~25 GB free disk space, internet access during installation |

## Quick start

Run the scripts in this folder by double-clicking them (or from a terminal). Each one writes a log to `windows_setup\logs\`.

| Script | What it does |
|---|---|
| `0_probe.bat` | Read-only check of GPU, CUDA Toolkit, Visual Studio, conda and git -> `logs\probe.log` |
| `1_install.bat` | Creates the `trellis` conda env, installs everything, builds the CUDA extensions, verifies them on the GPU, then runs the test below -> `logs\install.log`. Safe to re-run: finished steps are skipped |
| `2_run_test.bat` | Generates a 3D model from `assets\example_image\typical_creature_dragon.png`. Drag your own image(s) onto it to convert those instead -> `logs\test.log` |
| `3_seed_sweep.bat` | Drag an image onto it to generate 4 variations (seeds 1-4, 25 sampling steps) and keep the best |

Results go to `outputs\<image name>\` in the repo root:

- `<name>.glb` - textured mesh (glTF binary; imports directly into Blender, Unreal, Unity)
- `<name>.obj` + `.mtl` + `material_0.png` - the same mesh as OBJ
- `<name>_gaussians.ply` - 3D Gaussian splats
- `<name>_turntable.mp4`, `<name>_views.png`, `<name>_input_processed.png` - previews

Command-line equivalent (from a PowerShell prompt in the repo root):

```powershell
powershell -ExecutionPolicy Bypass -File windows_setup\install.ps1
powershell -ExecutionPolicy Bypass -File windows_setup\run_test.ps1 path\to\image1.png path\to\image2.jpg
```

`run_test.py` also reads `TRELLIS_SEEDS` (e.g. `1 2 3 4`, one output folder per seed) and `TRELLIS_STEPS`
(sampling steps for both stages, default 12).

## What gets installed

- Conda env `trellis`: Python 3.10 (conda-forge), PyTorch 2.6.0 + CUDA 12.6, torchvision 0.21.0
- `xformers` 0.0.29.post3 as the attention backend (`ATTN_BACKEND=xformers`; flash-attn has no official Windows wheels)
- `spconv-cu120` for sparse convolutions (`SPCONV_ALGO=native`)
- The packages from `setup.sh --basic` (pinned with `numpy<2`, see `constraints.txt`) and `utils3d` at the commit `setup.sh` uses
- Built from source with VS 2022 + CUDA 12.6 in `windows_setup\ext\`: [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
  and the [mip-splatting](https://github.com/autonomousvision/mip-splatting) `diff-gaussian-rasterization`.
  They are compiled for the GPU(s) reported by `nvidia-smi`; set `TORCH_CUDA_ARCH_LIST` (e.g. `7.5;8.6;8.9+PTX`)
  before running `install.ps1` to target other GPUs
- The `flexicubes` git submodule is initialised if it is missing

Nothing is compiled at run time, so `run_test.ps1` needs neither Visual Studio nor the CUDA Toolkit.

## Differences from the Linux setup

- **kaolin** is not installed (no Windows wheels). FlexiCubes only uses kaolin's `check_tensor` for input
  assertions, so `trellis/representations/mesh/cube2mesh.py` registers a small stand-in when kaolin is missing.
- **diffoctreerast** is not built, so radiance-field outputs cannot be rendered: `example.py`'s
  `sample_rf.mp4` step fails. Gaussians, meshes, GLB/OBJ/PLY export all work.
- The web demo (`app.py`, Gradio) is not set up by these scripts.

## Troubleshooting

- Look at the last lines of the matching log in `windows_setup\logs\`.
- `No module named 'triton'` from xformers is a harmless warning.
- `CERTIFICATE_VERIFY_FAILED` while downloading DINOv2: `run_test.py` points `SSL_CERT_FILE` at certifi's bundle; do the
  same in your own scripts if needed.
- Hugging Face symlink warning: harmless; enable Windows Developer Mode to save disk space in the model cache.
- Different CUDA Toolkit: change `$CudaVer` in `common.ps1` and the PyTorch/xformers wheels in `install.ps1` to a matching set.
- CUDA out of memory during GLB export: lower `texture_size`, or close other GPU applications.

## Licences

The two compiled rendering libraries (nvdiffrast, diff-gaussian-rasterization) are the same ones the Linux `setup.sh`
installs and are licensed for **non-commercial research and evaluation** only. See their repositories for details.
