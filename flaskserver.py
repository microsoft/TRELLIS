import os
import uuid
import shutil
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
from easydict import EasyDict as edict
import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

MAX_SEED = np.iinfo(np.int32).max

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

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
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

def preprocess_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    processed_image = pipeline.preprocess_image(image)
    return processed_image

@app.route('/generate_from_single_image', methods=['POST'])
def generate_from_single_image():
    try:
        # Get parameters
        seed = int(request.form.get('seed', 0))
        ss_guidance_strength = float(request.form.get('ss_guidance_strength', 7.5))
        ss_sampling_steps = int(request.form.get('ss_sampling_steps', 12))
        slat_guidance_strength = float(request.form.get('slat_guidance_strength', 3.0))
        slat_sampling_steps = int(request.form.get('slat_sampling_steps', 12))
        
        # Handle file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Create session directory
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        image_path = os.path.join(session_dir, filename)
        file.save(image_path)
        
        # Preprocess image
        image = preprocess_image(image_path)
        
        # Generate 3D model
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
        
        # Create output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save video preview
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        video_path = os.path.join(output_dir, 'preview.mp4')
        imageio.mimsave(video_path, video, fps=15)
        
        # Save state
        state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
        state_path = os.path.join(output_dir, 'state.pkl')
        torch.save(state, state_path)
        
        torch.cuda.empty_cache()
        
        return jsonify({
            'session_id': session_id,
            'preview_url': f'/download/preview/{session_id}',
            'state_url': f'/download/state/{session_id}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_from_multiple_images', methods=['POST'])
def generate_from_multiple_images():
    try:
        # Get parameters
        seed = int(request.form.get('seed', 0))
        ss_guidance_strength = float(request.form.get('ss_guidance_strength', 7.5))
        ss_sampling_steps = int(request.form.get('ss_sampling_steps', 12))
        slat_guidance_strength = float(request.form.get('slat_guidance_strength', 3.0))
        slat_sampling_steps = int(request.form.get('slat_sampling_steps', 12))
        multiimage_algo = request.form.get('multiimage_algo', 'stochastic')
        
        # Handle file uploads
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
            
        files = request.files.getlist('images')
        if len(files) == 0:
            return jsonify({'error': 'No selected files'}), 400
            
        # Create session directory
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded files
        images = []
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            image_path = os.path.join(session_dir, filename)
            file.save(image_path)
            images.append(Image.open(image_path))
        
        # Generate 3D model
        outputs = pipeline.run_multi_image(
            images,
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
        
        # Create output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save video preview
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        video_path = os.path.join(output_dir, 'preview.mp4')
        imageio.mimsave(video_path, video, fps=15)
        
        # Save state
        state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
        state_path = os.path.join(output_dir, 'state.pkl')
        torch.save(state, state_path)
        
        torch.cuda.empty_cache()
        
        return jsonify({
            'session_id': session_id,
            'preview_url': f'/download/preview/{session_id}',
            'state_url': f'/download/state/{session_id}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract_glb', methods=['POST'])
def extract_glb():
    try:
        # Get parameters
        session_id = request.form.get('session_id')
        mesh_simplify = float(request.form.get('mesh_simplify', 0.95))
        texture_size = int(request.form.get('texture_size', 1024))
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
            
        # Load state
        state_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'state.pkl')
        if not os.path.exists(state_path):
            return jsonify({'error': 'Invalid session_id'}), 404
            
        state = torch.load(state_path)
        gs, mesh = unpack_state(state)
        
        # Extract GLB
        glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
        
        # Save GLB
        glb_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'model.glb')
        glb.export(glb_path)
        
        torch.cuda.empty_cache()
        
        return jsonify({
            'glb_url': f'/download/glb/{session_id}'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/preview/<session_id>', methods=['GET'])
def download_preview(session_id):
    preview_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'preview.mp4')
    if not os.path.exists(preview_path):
        return jsonify({'error': 'Preview not found'}), 404
    return send_file(preview_path, as_attachment=True)

@app.route('/download/state/<session_id>', methods=['GET'])
def download_state(session_id):
    state_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'state.pkl')
    if not os.path.exists(state_path):
        return jsonify({'error': 'State not found'}), 404
    return send_file(state_path, as_attachment=True)

@app.route('/download/glb/<session_id>', methods=['GET'])
def download_glb(session_id):
    glb_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'model.glb')
    if not os.path.exists(glb_path):
        return jsonify({'error': 'GLB not found'}), 404
    return send_file(glb_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)