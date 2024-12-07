import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
from typing import *
import torch
import numpy as np
import imageio
import uuid
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/Trellis-demo"

os.makedirs(TMP_DIR, exist_ok=True)


def preprocess_image(image: Image.Image) -> Tuple[str, Image.Image]:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        str: uuid of the trial.
        Image.Image: The preprocessed image.
    """
    trial_id = str(uuid.uuid4())
    processed_image = pipeline.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image


def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
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
        'trial_id': trial_id,
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
    
    return gs, mesh, state['trial_id']


def image_to_3d(trial_id: str, seed: int, randomize_seed: bool, ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float, slat_sampling_steps: int) -> Tuple[dict, str]:
    """
    Convert an image to a 3D model.

    Args:
        trial_id (str): The uuid of the trial.
        seed (int): The random seed.
        randomize_seed (bool): Whether to randomize the seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    outputs = pipeline.run(
        Image.open(f"{TMP_DIR}/{trial_id}.png"),
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
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    trial_id = uuid.uuid4()
    video_path = f"{TMP_DIR}/{trial_id}.mp4"
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], trial_id)
    return state, video_path


def extract_glb(state: dict, mesh_simplify: float, texture_size: int) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    gs, mesh, trial_id = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = f"{TMP_DIR}/{trial_id}.glb"
    glb.export(glb_path)
    return glb_path, glb_path


def activate_button() -> gr.Button:
    return gr.Button(interactive=True)


def deactivate_button() -> gr.Button:
    return gr.Button(interactive=False)


with gr.Blocks() as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Upload an image and click "Generate" to create a 3D asset. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
    * If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
    """)
    
    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil", height=300)
            
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

            generate_btn = gr.Button("Generate")
            
            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
            
            extract_glb_btn = gr.Button("Extract GLB", interactive=False)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB", exposure=20.0, height=300)
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
            
    trial_id = gr.Textbox(visible=False)
    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row():
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[trial_id, image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )

    # Handlers
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[trial_id, image_prompt],
    )
    image_prompt.clear(
        lambda: '',
        outputs=[trial_id],
    )

    generate_btn.click(
        image_to_3d,
        inputs=[trial_id, seed, randomize_seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps],
        outputs=[output_buf, video_output],
    ).then(
        activate_button,
        outputs=[extract_glb_btn],
    )

    video_output.clear(
        deactivate_button,
        outputs=[extract_glb_btn],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        activate_button,
        outputs=[download_glb],
    )

    model_output.clear(
        deactivate_button,
        outputs=[download_glb],
    )
    

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    demo.launch(server_name="0.0.0.0")
