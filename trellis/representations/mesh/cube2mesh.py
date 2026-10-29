import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *


def _ensure_kaolin_check_tensor():
    """FlexiCubes imports ``kaolin.utils.testing.check_tensor``, but only uses it for input
    shape assertions. kaolin has no wheels for some platforms (e.g. Windows), so register a
    minimal stand-in module when the real package is not installed."""
    try:
        import kaolin.utils.testing  # noqa: F401
        return
    except ImportError:
        pass
    import sys
    import types

    def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
        ok = True
        if shape is not None:
            ok = len(shape) == tensor.ndim and all(d is None or d == tensor.shape[i] for i, d in enumerate(shape))
        if ok and dtype is not None:
            ok = tensor.dtype == dtype
        if ok and device is not None:
            ok = tensor.device.type == str(device).split(':')[0]
        if not ok and throw:
            raise ValueError(f"tensor check failed: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                             f"device={tensor.device} (expected shape={shape}, dtype={dtype}, device={device})")
        return ok

    testing = types.ModuleType('kaolin.utils.testing')
    testing.check_tensor = check_tensor
    sys.modules.setdefault('kaolin', types.ModuleType('kaolin'))
    sys.modules.setdefault('kaolin.utils', types.ModuleType('kaolin.utils'))
    sys.modules.setdefault('kaolin.utils.testing', testing)


_ensure_kaolin_check_tensor()
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        '''
        a model to generate a mesh from sparse features structures using flexicube
        '''
        super().__init__()
        self.device=device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
        self.use_color = use_color
        self._calc_layout()
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            '''
            6 channel color including normal map
            '''
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats : torch.Tensor, name : str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats : SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
        Returns:
            return the success tag and ni you loss, 
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
