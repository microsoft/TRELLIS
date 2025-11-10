<img src="assets/logo.webp" width="100%" align="center">
<h1 align="center">Structured 3D Latents<br>for Scalable and Versatile 3D Generation</h1>
<p align="center"><a href="https://arxiv.org/abs/2412.01506"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://microsoft.github.io/TRELLIS/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/Microsoft/TRELLIS'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>
<p align="center"><img src="assets/teaser.png" width="100%"></p>

<span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> is a large 3D asset generation model. It takes in text or image prompts and generates high-quality 3D assets in various formats, such as Radiance Fields, 3D Gaussians, and meshes. The cornerstone of <span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> is a unified Structured LATent (<span style="font-size: 16px; font-weight: 600;">SL</span><span style="font-size: 12px; font-weight: 700;">AT</span>) representation that allows decoding to different output formats and Rectified Flow Transformers tailored for <span style="font-size: 16px; font-weight: 600;">SL</span><span style="font-size: 12px; font-weight: 700;">AT</span> as the powerful backbones. We provide large-scale pre-trained models with up to 2 billion parameters on a large 3D asset dataset of 500K diverse objects. <span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> significantly surpasses existing methods, including recent ones at similar scales, and showcases flexible output format selection and local 3D editing capabilities which were not offered by previous models.

***Check out our [Project Page](https://microsoft.github.io/TRELLIS/) for more videos and interactive demos!***

<!-- Features -->
## 🌟 Features
- **High Quality**: It produces diverse 3D assets at high quality with intricate shape and texture details.
- **Versatility**: It takes text or image prompts and can generate various final 3D representations including but not limited to *Radiance Fields*, *3D Gaussians*, and *meshes*, accommodating diverse downstream requirements.
- **Flexible Editing**: It allows for easy editings of generated 3D assets, such as generating variants of the same object or local editing of the 3D asset.

<!-- Updates -->
## ⏩ Updates

**03/25/2025**
- Release training code.
- Release **TRELLIS-text** models and asset variants generation.
  - Examples are provided as [example_text.py](example_text.py) and [example_variant.py](example_variant.py).
  - Gradio demo is provided as [app_text.py](app_text.py).
  - *Note: It is always recommended to do text to 3D generation by first generating images using text-to-image models and then using TRELLIS-image models for 3D generation. Text-conditioned models are less creative and detailed due to data limitations.*

**12/26/2024**
- Release [**TRELLIS-500K**](https://github.com/microsoft/TRELLIS#-dataset) dataset and toolkits for data preparation.

**12/18/2024**
- Implementation of multi-image conditioning for **TRELLIS-image** model. ([#7](https://github.com/microsoft/TRELLIS/issues/7)). This is based on tuning-free algorithm without training a specialized model, so it may not give the best results for all input images.
- Add Gaussian export in `app.py` and `example.py`. ([#40](https://github.com/microsoft/TRELLIS/issues/40))

<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.  For windows setup, you may refer to [#3](https://github.com/microsoft/TRELLIS/issues/3) (not fully tested).
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A100 and A6000 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
    cd TRELLIS
    ```

2. Install the dependencies:
    
    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `trellis` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default the `trellis` environment will use pytorch 2.4.0 with CUDA 11.8. If you want to use a different version of CUDA (e.g., if you have CUDA Toolkit 12.2 installed and do not want to install another 11.8 version for submodule compilation), you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `PATH` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 11.8 and 12.2 installed, you should run `export PATH=/usr/local/cuda-11.8/bin:$PATH` and `export CUDA_HOME=/usr/local/cuda-11.8` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs do not support `flash-attn` (e.g., NVIDIA V100), you can remove the `--flash-attn` flag to install `xformers` only and set the `ATTN_BACKEND` environment variable to `xformers` before running the code. See the [Minimal Example](#minimal-example) for more details.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue or contact us.
    
    Create a new conda environment named `trellis` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --train                 Install training dependencies
        --xformers              Install xformers
        --flash-attn            Install flash-attn
        --diffoctreerast        Install diffoctreerast
        --spconv                Install spconv
        --mipgaussian           Install mip-splatting
        --kaolin                Install kaolin
        --nvdiffrast            Install nvdiffrast
        --demo                  Install all dependencies for demo
    ```

<!-- Pretrained Models -->
## 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| TRELLIS-image-large | Large image-to-3D model | 1.2B | [Download](https://huggingface.co/microsoft/TRELLIS-image-large) |
| TRELLIS-text-base | Base text-to-3D model | 342M | [Download](https://huggingface.co/microsoft/TRELLIS-text-base) |
| TRELLIS-text-large | Large text-to-3D model | 1.1B | [Download](https://huggingface.co/microsoft/TRELLIS-text-large) |
| TRELLIS-text-xlarge | Extra-large text-to-3D model | 2.0B | [Download](https://huggingface.co/microsoft/TRELLIS-text-xlarge) |

*Note: It is always recommended to use the image conditioned version of the models for better performance.*

*Note: All VAEs are included in **TRELLIS-image-large** model repo.*

The models are hosted on Hugging Face. You can directly load the models with their repository names in the code:
```python
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```

If you prefer loading the model from local, you can download the model files from the links above and load the model with the folder path (folder structure should be maintained):
```python
TrellisImageTo3DPipeline.from_pretrained("/path/to/TRELLIS-image-large")
```

<!-- Usage -->
## 💡 Usage

### Minimal Example

Here is an [example](example.py) of how to use the pretrained models for 3D asset generation.

```python
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("assets/example_image/T.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
```

After running the code, you will get the following files:
- `sample_gs.mp4`: a video showing the 3D Gaussian representation
- `sample_rf.mp4`: a video showing the Radiance Field representation
- `sample_mesh.mp4`: a video showing the mesh representation
- `sample.glb`: a GLB file containing the extracted textured mesh
- `sample.ply`: a PLY file containing the 3D Gaussian representation


### Web Demo

[app.py](app.py) provides a simple web demo for 3D asset generation. Since this demo is based on [Gradio](https://gradio.app/), additional dependencies are required:
```sh
. ./setup.sh --demo
```

After installing the dependencies, you can run the demo with the following command:
```sh
python app.py
```

Then, you can access the demo at the address shown in the terminal.


<!-- Dataset -->
## 📚 Dataset

We provide **TRELLIS-500K**, a large-scale dataset containing 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details.


<!-- Training -->
## 🏋️‍♂️ Training

TRELLIS’s training framework is organized to provide a flexible and modular approach to building and fine-tuning large-scale 3D generation models. The training code is centered around `train.py` and is structured into several directories to clearly separate dataset handling, model components, training logic, and visualization utilities.

### Code Structure

- **train.py**: Main entry point for training.
- **trellis/datasets**: Dataset loading and preprocessing.
- **trellis/models**: Different models and their components.
- **trellis/modules**: Custom modules for various models.
- **trellis/pipelines**: Inference pipelines for different models.
- **trellis/renderers**: Renderers for different 3D representations.
- **trellis/representations**: Different 3D representations.
- **trellis/trainers**: Training logic for different models.
- **trellis/utils**: Utility functions for training and visualization.

### Training Setup

1. **Prepare the Environment:**
   - Ensure all training dependencies are installed.
   - Use a Linux system with an NVIDIA GPU (The models are trained on NVIDIA A100 GPUs).
   - For distributed training, verify that your nodes can communicate through the designated master address and port.

2. **Dataset Preparation:**
   - Organize your dataset similar to TRELLIS-500K. Specify your dataset path using the `--data_dir` argument when launching training.

3. **Configuration Files:**
   - Training hyperparameters and model architectures are defined in configuration files under the `configs/` directory.
   - Example configuration files include:

| Config | Pretained Model | Description |
| --- | --- | --- |
| [`vae/ss_vae_conv3d_16l8_fp16.json`](configs/vae/ss_vae_conv3d_16l8_fp16.json) | [Encoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/ss_enc_conv3d_16l8_fp16.safetensors) [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors) | Sparse structure VAE |
| [`vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json) | [Encoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors) [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors) | SLat VAE with Gaussian Decoder |
| [`vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json) | [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors) | SLat Radiance Field Decoder |
| [`vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json`](configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json) | [Decoder](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors) | SLat Mesh Decoder |
| [`generation/ss_flow_img_dit_L_16l8_fp16.json`](configs/generation/ss_flow_img_dit_L_16l8_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors) | Image conditioned sparse structure Flow Model |
| [`generation/slat_flow_img_dit_L_64l8p2_fp16.json`](configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-image-large/blob/main/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors) | Image conditioned SLat Flow Model |
| [`generation/ss_flow_txt_dit_B_16l8_fp16.json`](configs/generation/ss_flow_txt_dit_B_16l8_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-base/blob/main/ckpts/ss_flow_txt_dit_B_16l8_fp16.safetensors) | Base text-conditioned sparse structure Flow Model |
| [`generation/slat_flow_txt_dit_B_64l8p2_fp16.json`](configs/generation/slat_flow_txt_dit_B_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-base/blob/main/ckpts/slat_flow_txt_dit_B_64l8p2_fp16.safetensors) | Base text-conditioned SLat Flow Model |
| [`generation/ss_flow_txt_dit_L_16l8_fp16.json`](configs/generation/ss_flow_txt_dit_L_16l8_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-large/blob/main/ckpts/ss_flow_txt_dit_L_16l8_fp16.safetensors) | Large text-conditioned sparse structure Flow Model |
| [`generation/slat_flow_txt_dit_L_64l8p2_fp16.json`](configs/generation/slat_flow_txt_dit_L_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-large/blob/main/ckpts/slat_flow_txt_dit_L_64l8p2_fp16.safetensors) | Large text-conditioned SLat Flow Model |
| [`generation/ss_flow_txt_dit_XL_16l8_fp16.json`](configs/generation/ss_flow_txt_dit_XL_16l8_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-xlarge/blob/main/ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors) | Extra-large text-conditioned sparse structure Flow Model |
| [`generation/slat_flow_txt_dit_XL_64l8p2_fp16.json`](configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16.json) | [Denoiser](https://huggingface.co/microsoft/TRELLIS-text-xlarge/blob/main/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16.safetensors) | Extra-large text-conditioned SLat Flow Model |

### Command-Line Options

The training script can be run as follows:
```sh
usage: train.py [-h] --config CONFIG --output_dir OUTPUT_DIR [--load_dir LOAD_DIR] [--ckpt CKPT] [--data_dir DATA_DIR] [--auto_retry AUTO_RETRY] [--tryrun] [--profile] [--num_nodes NUM_NODES] [--node_rank NODE_RANK] [--num_gpus NUM_GPUS] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]

options:
  -h, --help                    show this help message and exit
  --config CONFIG               Experiment config file
  --output_dir OUTPUT_DIR       Output directory
  --load_dir LOAD_DIR           Load directory, default to output_dir
  --ckpt CKPT                   Checkpoint step to resume training, default to latest
  --data_dir DATA_DIR           Data directory
  --auto_retry AUTO_RETRY       Number of retries on error
  --tryrun                      Try run without training
  --profile                     Profile training
  --num_nodes NUM_NODES         Number of nodes
  --node_rank NODE_RANK         Node rank
  --num_gpus NUM_GPUS           Number of GPUs per node, default to all
  --master_addr MASTER_ADDR     Master address for distributed training
  --master_port MASTER_PORT     Port for distributed training
```

### Example Training Commands

#### Single-node Training

To train a image-to-3D stage 2 model with a single machine.
```sh
python train.py \
  --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
  --output_dir outputs/slat_vae_dec_mesh_swin8_B_64l8_fp16_1node \
  --data_dir /path/to/your/dataset1,/path/to/your/dataset2 \
```
The script will automatically distribute the training across all available GPUs. Specify the number of GPUs with the `--num_gpus` flag if you want to limit the number of GPUs used.

#### Multi-node Training

To train a image-to-3D stage 2 model with multiple GPUs across nodes (e.g., 2 nodes):
```sh
python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_2nodes \
  --data_dir /path/to/your/dataset1,/path/to/your/dataset2 \
  --num_nodes 2 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
```
Be sure to adjust `node_rank`, `master_addr`, and `master_port` for each node accordingly.

#### Resuming Training

By default, training will resume from the latest saved checkpoint in the same output directory. To specify a specific checkpoint to resume from, use the `--load_dir` and `--ckpt` flags:
```sh
python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json \
  --output_dir outputs/slat_flow_img_dit_L_64l8p2_fp16_resume \
  --data_dir /path/to/your/dataset1,/path/to/your/dataset2 \
  --load_dir /path/to/your/checkpoint \
  --ckpt [step]
```

### Additional Options

- **Auto Retry:** Use the `--auto_retry` flag to specify the number of retries in case of intermittent errors.
- **Dry Run:** The `--tryrun` flag allows you to check your configuration and environment without launching full training.
- **Profiling:** Enable profiling with the `--profile` flag to gain insights into training performance and diagnose bottlenecks.

Adjust the file paths and parameters to match your experimental setup.


<!-- License -->
## ⚖️ License

TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE). The following submodules may have different licenses:
- [**diffoctreerast**](https://github.com/JeffreyXiang/diffoctreerast): We developed a CUDA-based real-time differentiable octree renderer for rendering radiance fields as part of this project. This renderer is derived from the [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) project and is available under the [LICENSE](https://github.com/JeffreyXiang/diffoctreerast/blob/master/LICENSE).


- [**Modified Flexicubes**](https://github.com/MaxtirError/FlexiCubes): In this project, we used a modified version of [Flexicubes](https://github.com/nv-tlabs/FlexiCubes) to support vertex attributes. This modified version is licensed under the [LICENSE](https://github.com/nv-tlabs/FlexiCubes/blob/main/LICENSE.txt).


<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```

