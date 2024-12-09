import os
from typing import *
import torch
import numpy as np
import imageio
import uuid
from easydict import EasyDict as edict
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

app = FastAPI()

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/Trellis-demo"
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize pipeline globally
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

def preprocess_image(image: Image.Image) -> Tuple[str, Image.Image]:
    trial_id = str(uuid.uuid4())
    processed_image = pipeline.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image

def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy().tolist(),
            '_features_dc': gs._features_dc.cpu().numpy().tolist(),
            '_scaling': gs._scaling.cpu().numpy().tolist(),
            '_rotation': gs._rotation.cpu().numpy().tolist(),
            '_opacity': gs._opacity.cpu().numpy().tolist(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy().tolist(),
            'faces': mesh.faces.cpu().numpy().tolist(),
        },
        'trial_id': trial_id,
    }

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    seed: int = 0,
    randomize_seed: bool = True,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12
):
    # Read and process the uploaded image
    image = Image.open(file.file)
    trial_id, processed_image = preprocess_image(image)
    
    # Generate 3D model
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    
    outputs = pipeline.run(
        processed_image,
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
    
    # Generate preview video
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = f"{TMP_DIR}/{trial_id}_preview.mp4"
    imageio.mimsave(video_path, video, fps=15)
    
    # Pack state and return results
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], trial_id)
    
    return {
        "trial_id": trial_id,
        "state": state,
        "preview_video": f"/preview/{trial_id}"
    }

@app.get("/preview/{trial_id}")
async def get_preview(trial_id: str):
    video_path = f"{TMP_DIR}/{trial_id}_preview.mp4"
    return FileResponse(video_path)

@app.post("/extract-glb/{trial_id}")
async def extract_glb(
    trial_id: str,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024
):
    # Load the state file
    state_path = f"{TMP_DIR}/{trial_id}_state.json"
    if not os.path.exists(state_path):
        return {"error": "Trial ID not found"}
    
    # Generate GLB
    glb_path = f"{TMP_DIR}/{trial_id}.glb"
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
    
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(glb_path)
    
    return FileResponse(glb_path, filename=f"{trial_id}.glb")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
