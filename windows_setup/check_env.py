"""Smoke-test every TRELLIS dependency on the GPU."""
import os, sys, traceback
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

results = []
def check(name, fn):
    try:
        r = fn()
        print(f'[OK]   {name}: {r}', flush=True)
        results.append((name, True))
    except Exception as e:
        print(f'[FAIL] {name}: {e!r}', flush=True)
        traceback.print_exc()
        results.append((name, False))

import torch
check('torch', lambda: (torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)))

def t_xformers():
    import xformers, xformers.ops as xops
    q = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.float16)
    return (xformers.__version__, tuple(xops.memory_efficient_attention(q, q, q).shape))
check('xformers', t_xformers)

def t_spconv():
    import spconv, spconv.pytorch as sp
    g = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2), indexing='ij'), -1).reshape(-1, 3)
    idx = torch.cat([torch.zeros(8, 1, dtype=torch.int64), g], 1).int().cuda()
    x = sp.SparseConvTensor(torch.randn(8, 4, device='cuda'), idx, [4, 4, 4], 1)
    conv = sp.SubMConv3d(4, 8, 3, algo=sp.ConvAlgo.Native).cuda()
    return (spconv.__version__, tuple(conv(x).features.shape))
check('spconv', t_spconv)

def t_nvdiffrast():
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    pos = torch.tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [0, 0.8, 0, 1]]], device='cuda', dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], device='cuda', dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
    return ('covered px', int((rast[..., 3] > 0).sum()))
check('nvdiffrast', t_nvdiffrast)

def t_dgr():
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    import inspect
    return 'kernel_size' in inspect.signature(GaussianRasterizationSettings).parameters or str(GaussianRasterizationSettings._fields)
check('diff_gaussian_rasterization', t_dgr)

def t_meshfix():
    from pymeshfix import _meshfix
    _meshfix.PyTMesh()
    import pymeshfix
    return getattr(pymeshfix, '__version__', '?')
check('pymeshfix', t_meshfix)

for mod in ['utils3d', 'xatlas', 'pyvista', 'igraph', 'trimesh', 'rembg', 'cv2', 'imageio_ffmpeg', 'open3d', 'transformers', 'easydict']:
    check(mod, lambda m=mod: __import__(m).__dict__.get('__version__', 'imported'))

def t_trellis():
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    from trellis.representations.mesh.flexicubes.flexicubes import FlexiCubes
    return 'imported'
check('trellis', t_trellis)

bad = [n for n, ok in results if not ok]
print('\nSUMMARY:', 'ALL OK' if not bad else 'FAILED: ' + ', '.join(bad), flush=True)
sys.exit(1 if bad else 0)
