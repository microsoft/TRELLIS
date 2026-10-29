"""TRELLIS image-to-3D test run: one image -> textured GLB + OBJ (+MTL/PNG), Gaussian PLY, turntable MP4."""
import os, sys, time, argparse
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('TQDM_MININTERVAL', '3')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

# torch.hub (DINOv2 code + weights) downloads through urllib, which uses the conda env's OpenSSL CA store;
# on some machines that store fails certificate checks. Use certifi's bundle, as huggingface_hub does.
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class _CRtoLF:
    """Turn tqdm's carriage-return updates into new lines so progress shows up live in the log."""
    def __init__(self, f): self._f = f
    def write(self, s): return self._f.write(s.replace('\r', '\n'))
    def flush(self): self._f.flush()
    def __getattr__(self, n): return getattr(self._f, n)


sys.stderr = _CRtoLF(sys.stderr)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process(pipeline, image_path, args, seed, tag=''):
    import numpy as np
    import torch
    import imageio
    from PIL import Image
    from trellis.utils import render_utils, postprocessing_utils

    stem = os.path.splitext(os.path.basename(image_path))[0] + tag
    out_dir = os.path.join(args.out, stem)
    os.makedirs(out_dir, exist_ok=True)
    log(f'=== {image_path} | seed {seed}' + (f' | steps {args.steps}' if args.steps else ''))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    image = Image.open(image_path)
    log(f'Input size: {image.size[0]}x{image.size[1]} {image.mode}')
    pipeline.preprocess_image(image).save(os.path.join(out_dir, f'{stem}_input_processed.png'))

    kw = {}
    if args.steps:
        kw = dict(sparse_structure_sampler_params={'steps': args.steps}, slat_sampler_params={'steps': args.steps})
    outputs = pipeline.run(image, seed=seed, **kw)
    torch.cuda.synchronize()
    log(f'3D generation done in {time.time() - t0:.0f}s')
    gs, mesh = outputs['gaussian'][0], outputs['mesh'][0]

    frames = render_utils.render_video(gs, num_frames=120)['color']
    imageio.mimsave(os.path.join(out_dir, f'{stem}_turntable.mp4'), frames, fps=30)
    imageio.imwrite(os.path.join(out_dir, f'{stem}_views.png'), np.concatenate([frames[i] for i in (0, 30, 60, 90)], axis=1))
    gs.save_ply(os.path.join(out_dir, f'{stem}_gaussians.ply'))
    log('Turntable video + Gaussian PLY saved')

    t3 = time.time()
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=args.simplify, texture_size=args.texture_size)
    glb_path = os.path.join(out_dir, f'{stem}.glb')
    glb.export(glb_path)
    log(f'GLB saved in {time.time() - t3:.0f}s: {glb_path} ({len(glb.vertices)} verts, {len(glb.faces)} tris)')

    obj_path = os.path.join(out_dir, f'{stem}.obj')
    try:
        from trimesh.exchange.obj import export_obj
        obj_mesh = glb.copy()
        mat = obj_mesh.visual.material
        if hasattr(mat, 'to_simple'):
            obj_mesh.visual.material = mat.to_simple()
        obj_text, extra = export_obj(obj_mesh, include_texture=True, return_texture=True, mtl_name=f'{stem}.mtl')
        with open(obj_path, 'w', encoding='utf-8') as f:
            f.write(obj_text)
        for name, data in extra.items():
            with open(os.path.join(out_dir, name), 'wb') as f:
                f.write(data)
        log(f'OBJ saved: {obj_path} (+ {", ".join(extra)})')
    except Exception as e:
        log(f'OBJ export with texture failed ({e!r}); writing plain OBJ')
        glb.export(obj_path)

    log(f'Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB | time for this image: {time.time() - t0:.0f}s')
    log(f'DONE -> {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*', help='one or more image files (default: the dragon example)')
    ap.add_argument('--seed', type=int, default=1)
    env_seeds = [int(x) for x in os.environ.get('TRELLIS_SEEDS', '').split()] or None
    env_steps = int(os.environ['TRELLIS_STEPS']) if os.environ.get('TRELLIS_STEPS') else None
    ap.add_argument('--seeds', type=int, nargs='+', default=env_seeds, help='several seeds -> one output folder per seed (or env TRELLIS_SEEDS)')
    ap.add_argument('--steps', type=int, default=env_steps, help='sampler steps for both stages, default 12 (or env TRELLIS_STEPS)')
    ap.add_argument('--texture-size', type=int, default=1024)
    ap.add_argument('--simplify', type=float, default=0.95, help='fraction of triangles to remove')
    ap.add_argument('--out', default=os.path.join(REPO, 'outputs'))
    args = ap.parse_args()
    images = args.images or [os.path.join(REPO, 'assets', 'example_image', 'typical_creature_dragon.png')]

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    log(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}')
    t0 = time.time()
    log('Loading microsoft/TRELLIS-image-large...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    pipeline.cuda()
    log(f'Pipeline ready in {time.time() - t0:.0f}s')

    seeds = args.seeds or [args.seed]
    jobs = [(p, sd) for p in images for sd in seeds]
    failed = []
    for path, sd in jobs:
        try:
            process(pipeline, path, args, sd, f'_seed{sd}' if args.seeds else '')
        except Exception:
            import traceback
            traceback.print_exc()
            log(f'FAILED: {path}')
            failed.append(f'{path} (seed {sd})')
            torch.cuda.empty_cache()

    log(f'Total time: {time.time() - t0:.0f}s | {len(jobs) - len(failed)}/{len(jobs)} succeeded')
    if failed:
        sys.exit(1)
    log('ALL DONE')


if __name__ == '__main__':
    main()
