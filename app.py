import os
import sys 
import subprocess
import torch
import numpy as np
import imageio
import shutil 
import gradio as gr
from enum import Enum 
from typing import *
from functools import partial 
from easydict import EasyDict as edict
from PIL import Image
from gradio_litmodel3d import LitModel3D
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils, general_utils

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

pipeline = None

# Default sampling steps for SparseStructure SLAT 
DEFAULT_SS_SAMPLE_STEPS = 12
DEFAULT_SLAT_SAMPLE_STEPS = 12
DETAIL_VAR_SLAT_SAMPLE_STEPS = 25 

# Voxelizer (for detail variation pipeline)
VOXELIZER = "cuda_voxelizer"
VOVEL_RESOLUTION = 64 
VOXELIZER_CMD = "{} -f {} -s {} -o binvox"

class DemoMode(Enum):
    SINGLE_IMAGE = 0
    MULTI_IMAGE = 1 
    DETAIL_VARIATION = 2 

def check_voxelizer_exists(executable):
    """ Install the CUDA Voxelizer according to https://github.com/Forceflow/cuda_voxelizer.
    """
    try:
        subprocess.check_output([executable])
    except subprocess.CalledProcessError:
        print("Can not find the voxelizer!!! Install it according to https://github.com/Forceflow/cuda_voxelizer.")
        sys.exit(-1)

def voxelize_mesh(inpath: str):
    """ 
    Voxelize given mesh(OBJ) 

    Args:
        inpath (str): input path of the mesh model 

    Returns:
        str: output path 
    """
    outpath = os.path.join(os.path.dirname(inpath), f"{os.path.basename(inpath)}_{VOVEL_RESOLUTION}.binvox")
    if os.path.exists(outpath):
        return outpath 
    
    check_voxelizer_exists(VOXELIZER)
    cmd = VOXELIZER_CMD.format(VOXELIZER, inpath, VOVEL_RESOLUTION)
    voxelizer_ans = subprocess.run(cmd, capture_output=True, shell=True)
    if voxelizer_ans.returncode != 0 or not os.path.exists(outpath):
        print("voxelizer fails with error:")
        print(voxelizer_ans.stderr)
    return outpath 

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    global pipeline 
    if pipeline is None:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()

    processed_image = pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images.
    
    Args:
        images (List[Tuple[Image.Image, str]]): The input images.
        
    Returns:
        List[Image.Image]: The preprocessed images.
    """
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

def preprocess_mesh(inpath: str) -> str: 
    """
    Preprocess a mesh for PREVIEW (colorize and rotation)
    
    Args:
        inpath (str): input path of the mesh 

    Returns:
        str: path of the processed mesh 
    """
    import trimesh 
    inmesh = trimesh.load(inpath)
    vertices = np.array(inmesh.vertices)
    # Normalize 
    colors = (vertices - vertices.min(axis=0, keepdims=True)) / (vertices.max(axis=0, keepdims=True) - vertices.min(axis=0, keepdims=True))
    colors = (colors * 255).astype(np.uint8)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array([[-1, 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    newmesh = trimesh.Trimesh(vertices, inmesh.faces, visual=trimesh.visual.color.ColorVisuals(vertex_colors=colors))
    outpath = os.path.join(os.path.dirname(inpath), os.path.splitext(os.path.basename(inpath))[0] + "_colorized.obj")
    newmesh.export(outpath)
    return outpath  

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    image_detail_var: Image.Image, 
    raw_mesh: str, 
    demo_mode: DemoMode,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert an image to a 3D model.

    Args:
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
        image_detail_var (Image.Image): The input image used for detailed variation.
        raw_mesh (str): raw mesh (ONLY required for Detail Variation mode).
        demo_mode (Demo): Mode of current run (SingleImageTo3D / MultiImageTo3D / DetailVariation).
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """    
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    
    if demo_mode == DemoMode.SINGLE_IMAGE:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    elif demo_mode == DemoMode.MULTI_IMAGE:
        outputs = pipeline.run_multi_image(
            [image[0] for image in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
    else:
        import trimesh 
        binary_voxel = trimesh.load(voxelize_mesh(raw_mesh)).matrix 
        outputs = pipeline.run_detail_variation(
            binary_voxel, 
            image_detail_var,
            seed=seed, 
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    torch.cuda.empty_cache()
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    return state, video_path


def extract_mesh(
    state: dict, 
    get_srgb_texture: bool, 
    texture_bake_mode: Literal['opt', 'fast'], 
    mesh_simplify: float, 
    texture_size: int, 
    req: gr.Request, 
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    extracted_mesh = postprocessing_utils.to_trimesh(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, texture_bake_mode=texture_bake_mode, get_srgb_texture=get_srgb_texture, verbose=True)
    glb_path = os.path.join(user_dir, "sample.glb")
    extracted_mesh.export(glb_path)
    # also export the obj version 
    obj_dir = os.path.join(user_dir, "sample_obj")
    if os.path.exists(obj_dir):
        shutil.rmtree(obj_dir)
    obj_zip_path = os.path.join(user_dir, "sample_obj.zip")
    os.makedirs(obj_dir, exist_ok=True)
    extracted_mesh.export(os.path.join(obj_dir, f"sample.obj"))
    # compress obj and its dependencies 
    general_utils.zip_directory(obj_dir, obj_zip_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path, obj_zip_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split an image into multiple views.
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Upload an image and click "Generate" to create a 3D asset. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
    * If you find the generated 3D asset satisfactory, click "Extract Mesh" to extract the GLB/OBJ file and download it.
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as input_tabs:
                with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                    image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)
                with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                    multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=300, columns=3)
                    gr.Markdown("""
                        Input different views of the object in separate images. 
                        
                        *NOTE: this is an experimental algorithm without training a specialized model. It may not produce the best results for all images, especially those having different poses or inconsistent details.*
                    """)
                with gr.Tab(label="Detail Variation", id=2) as detail_var_input_tab:
                    with gr.Row():
                        mesh_prompt = gr.Model3D(label="Input Raw Mesh(Z-UP)", height=300, display_mode="Solid", clear_color=[1, 1, 1, 1])
                        mesh_prompt_view = gr.Model3D(label="Preview Input Mesh", height=300, display_mode="Solid", clear_color=[1, 1, 1, 1])
                        image_prompt_detail_var = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)
                    gr.Markdown("""
                        Input a mesh w/o texture and a reference image. 
                        
                        *NOTE: this is an experimental feature.
                    """)
            
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=DEFAULT_SS_SAMPLE_STEPS, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=DEFAULT_SLAT_SAMPLE_STEPS, step=1)
                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic")

            generate_btn = gr.Button("Generate")
            
            with gr.Accordion(label="Mesh Extraction Settings", open=True):
                get_srgb_texture = gr.Checkbox(label="Get sRGB Texture", value=False)
                texture_bake_mode = gr.Radio(["opt", "fast"], label="Texture Bake Mode", value='opt')
                mesh_simplify = gr.Slider(0.5, 0.99, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
            
            with gr.Row():
                extract_mesh_btn = gr.Button("Extract Mesh", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("""
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=250)
            model_output = LitModel3D(label="Extracted Mesh / Gaussian", exposure=20.0, height=250)
            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_obj = gr.DownloadButton(label="Download OBJ", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)  
            
    # single_image / multi_image / detail variation 
    demo_mode = gr.State(DemoMode.SINGLE_IMAGE)
    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row() as single_image_example:
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )
    with gr.Row(visible=False) as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )
    with gr.Row(visible=False) as detail_var_example: 
        examples_mesh = gr.Examples(
            examples=[
                f'assets/example_mesh/{mesh}'
                for mesh in os.listdir("assets/example_mesh") if mesh.endswith(".obj")
            ],
            fn=preprocess_mesh,
            inputs=[mesh_prompt],
            outputs=[mesh_prompt_view],
            run_on_click=True,
            examples_per_page=64,
            label="Example Meshes"
        )
          
        examples_image = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt_detail_var],
            fn=preprocess_image,
            outputs=[image_prompt_detail_var],
            run_on_click=True,
            examples_per_page=64,
            label="Example Images"
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    single_image_input_tab.select(
        lambda: tuple([DemoMode.SINGLE_IMAGE, DEFAULT_SLAT_SAMPLE_STEPS, gr.Row.update(visible=True), gr.Row.update(visible=False), gr.Row.update(visible=False)]),
        outputs=[demo_mode, slat_sampling_steps, single_image_example, multiimage_example, detail_var_example]
    )
    multiimage_input_tab.select(
        lambda: tuple([DemoMode.MULTI_IMAGE, DEFAULT_SLAT_SAMPLE_STEPS, gr.Row.update(visible=False), gr.Row.update(visible=True), gr.Row.update(visible=False)]),
        outputs=[demo_mode, slat_sampling_steps, single_image_example, multiimage_example, detail_var_example]
    )
    detail_var_input_tab.select(
        lambda: tuple([DemoMode.DETAIL_VARIATION, DETAIL_VAR_SLAT_SAMPLE_STEPS, gr.Row.update(visible=False), gr.Row.update(visible=False), gr.Row.update(visible=True)]),
        outputs=[demo_mode, slat_sampling_steps, single_image_example, multiimage_example, detail_var_example] 
    )
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )
    image_prompt_detail_var.upload(
        preprocess_images, 
        inputs=[image_prompt_detail_var],
        outputs=[image_prompt_detail_var],
    )
    mesh_prompt.upload(
        preprocess_mesh, 
        inputs=[mesh_prompt], 
        outputs=[mesh_prompt_view]
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[image_prompt, multiimage_prompt, image_prompt_detail_var, mesh_prompt, demo_mode, seed, ss_guidance_strength,  ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_mesh_btn, extract_gs_btn],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_mesh_btn, extract_gs_btn],
    )
    
    extract_mesh_btn.click(
        extract_mesh,
        inputs=[output_buf, get_srgb_texture, texture_bake_mode, mesh_simplify, texture_size],
        outputs=[model_output, download_glb, download_obj],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[download_glb, download_obj],
    )
    
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[download_glb, download_obj],
    )
    
    

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(debug=True, server_name='0.0.0.0', server_port=6006)
