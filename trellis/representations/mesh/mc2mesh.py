import numpy as np
import torch
from easydict import EasyDict as edict
from skimage import measure
from typing import Tuple, Optional

from .cube2mesh import MeshExtractResult
from .utils_cube import *
from ...modules.sparse import SparseTensor


class EnhancedMarchingCubes:
    def __init__(self, device="cuda"):
        self.device = device

    def __call__(self,
                 voxelgrid_vertices: torch.Tensor,
                 scalar_field: torch.Tensor,
                 voxelgrid_colors: Optional[torch.Tensor] = None,
                 training: bool = False
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced Marching Cubes implementation that handles deformations and colors
        """
        if scalar_field.dim() == 1:
            grid_size = int(round(scalar_field.shape[0] ** (1 / 3)))
            scalar_field = scalar_field.reshape(grid_size, grid_size, grid_size)
        elif scalar_field.dim() > 3:
            scalar_field = scalar_field.squeeze()

        # Convert to numpy and ensure values are in correct range
        scalar_np = scalar_field.cpu().numpy()

        if scalar_np.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {scalar_np.shape}")

        # Run marching cubes with normalized coordinates
        vertices, faces, normals, _ = measure.marching_cubes(
            scalar_np,
            level=0.0,
            gradient_direction='ascent'
        )

        vertices = torch.from_numpy(np.ascontiguousarray(vertices)).float().to(self.device)
        faces = torch.from_numpy(np.ascontiguousarray(faces)).long().to(self.device)

        # Apply deformations
        if voxelgrid_vertices is not None:
            # Reshape and normalize voxelgrid_vertices if needed
            if voxelgrid_vertices.dim() == 2:
                voxelgrid_vertices = voxelgrid_vertices.reshape(grid_size, grid_size, grid_size, 3)
            deformed_vertices = self._apply_deformations(vertices, voxelgrid_vertices)
        else:
            deformed_vertices = vertices

        # Handle colors if provided
        colors = None
        if voxelgrid_colors is not None:
            if voxelgrid_colors.dim() == 2:
                voxelgrid_colors = voxelgrid_colors.reshape(grid_size, grid_size, grid_size, -1)
            colors = self._interpolate_colors(vertices, voxelgrid_colors)
            # Ensure colors are in [0, 1] range
            colors = torch.sigmoid(colors)

        # Compute deviation loss for training
        deviation_loss = torch.tensor(0.0, device=self.device)
        if training:
            deviation_loss = self._compute_deviation_loss(vertices, deformed_vertices)

        faces = faces.flip(dims=[1])  # Reverse the order of vertices in each face, for some reason it's reversed...

        return deformed_vertices, faces, deviation_loss, colors

    def _apply_deformations(self, vertices: torch.Tensor,
                            voxelgrid_vertices: torch.Tensor) -> torch.Tensor:
        """Apply deformations to vertices using trilinear interpolation"""

        grid_positions = vertices.clone()

        # Scale to grid coordinates
        grid_coords = grid_positions.long()
        local_coords = grid_positions - grid_coords.float()

        # Reshape voxelgrid_vertices if needed
        if voxelgrid_vertices.dim() == 2:
            # Assuming voxelgrid_vertices is [N, 3]
            grid_size = int(round(voxelgrid_vertices.shape[0] ** (1 / 3)))
            voxelgrid_vertices = voxelgrid_vertices.reshape(grid_size, grid_size, grid_size, 3)

        # Ensure coordinates are within bounds
        grid_coords = torch.clamp(grid_coords, 0, voxelgrid_vertices.shape[0] - 1)

        # Perform trilinear interpolation
        deformed_vertices = self._trilinear_interpolate(
            grid_coords, local_coords, voxelgrid_vertices
        )

        return deformed_vertices

    def _interpolate_colors(self, vertices: torch.Tensor,
                            voxelgrid_colors: torch.Tensor) -> torch.Tensor:
        """Interpolate colors for vertices"""

        # Get grid positions
        grid_positions = vertices.clone()

        # Scale to grid coordinates
        grid_coords = grid_positions.long()
        local_coords = grid_positions - grid_coords.float()

        # Reshape colors if they're in 2D format
        if voxelgrid_colors.dim() == 2:
            grid_size = int(round(voxelgrid_colors.shape[0] ** (1 / 3)))
            color_channels = voxelgrid_colors.shape[1]
            voxelgrid_colors = voxelgrid_colors.reshape(grid_size, grid_size, grid_size, color_channels)

        # Ensure coordinates are within bounds
        grid_coords = torch.clamp(grid_coords, 0, voxelgrid_colors.shape[0] - 1)

        # Perform trilinear interpolation
        return self._trilinear_interpolate(
            grid_coords, local_coords, voxelgrid_colors, is_color=True
        )

    def _trilinear_interpolate(self, grid_coords: torch.Tensor,
                               local_coords: torch.Tensor,
                               values: torch.Tensor,
                               is_color: bool = False) -> torch.Tensor:
        """Perform trilinear interpolation"""
        x, y, z = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]

        if is_color and values.dim() == 2:
            # Handle flat color array
            grid_size = int(round(values.shape[0] ** (1 / 3)))
            color_channels = values.shape[1]
            values = values.reshape(grid_size, grid_size, grid_size, color_channels)

        # Get corner values with proper indexing based on dimensionality
        if values.dim() == 4:  # For 4D tensors (grid x grid x grid x channels)
            c000 = values[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], :]
            c001 = values[grid_coords[:, 0], grid_coords[:, 1],
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c010 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   grid_coords[:, 2], :]
            c011 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c100 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1],
                   grid_coords[:, 2], :]
            c101 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1],
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
            c110 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1),
                   torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), grid_coords[:, 2], :]
            c111 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1),
                   torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1),
                   torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1), :]
        else:
            c000 = values[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]
            c001 = values[
                grid_coords[:, 0], grid_coords[:, 1], torch.clamp(grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c010 = values[
                grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), grid_coords[:, 2]]
            c011 = values[grid_coords[:, 0], torch.clamp(grid_coords[:, 1] + 1, 0, values.shape[1] - 1), torch.clamp(
                grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c100 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1], grid_coords[:, 2]]
            c101 = values[torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), grid_coords[:, 1], torch.clamp(
                grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]
            c110 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), torch.clamp(grid_coords[:, 1] + 1, 0,
                                                                                        values.shape[
                                                                                            1] - 1), grid_coords[:, 2]]
            c111 = values[
                torch.clamp(grid_coords[:, 0] + 1, 0, values.shape[0] - 1), torch.clamp(grid_coords[:, 1] + 1, 0,
                                                                                        values.shape[
                                                                                            1] - 1), torch.clamp(
                    grid_coords[:, 2] + 1, 0, values.shape[2] - 1)]

        # Add channel dimension for 3D tensors if needed
        if values.dim() == 3:
            c000, c001, c010, c011 = [c[..., None] if c.dim() == 1 else c for c in [c000, c001, c010, c011]]
            c100, c101, c110, c111 = [c[..., None] if c.dim() == 1 else c for c in [c100, c101, c110, c111]]

        # Interpolate along x
        c00 = c000 * (1 - x)[:, None] + c100 * x[:, None]
        c01 = c001 * (1 - x)[:, None] + c101 * x[:, None]
        c10 = c010 * (1 - x)[:, None] + c110 * x[:, None]
        c11 = c011 * (1 - x)[:, None] + c111 * x[:, None]

        # Interpolate along y
        c0 = c00 * (1 - y)[:, None] + c10 * y[:, None]
        c1 = c01 * (1 - y)[:, None] + c11 * y[:, None]

        # Interpolate along z
        return c0 * (1 - z)[:, None] + c1 * z[:, None]

    def _compute_deviation_loss(self, original_vertices: torch.Tensor,
                                deformed_vertices: torch.Tensor) -> torch.Tensor:
        """Compute deviation loss for training"""
        return torch.mean((deformed_vertices - original_vertices) ** 2)


class SparseFeatures2MCMesh:
    def __init__(self, device="cuda", res=128, use_color=True):
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = EnhancedMarchingCubes(device=device)
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

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name][
            'shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats

        sdf, deform, color, weights = [self.get_layout(feats, name)
                                       for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1),
                                                     training=training)

        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res + 1, sdf_init=True)

        if self.use_color:
            sdf_d, deform_d, colors_d = (v_attrs_d[..., 0], v_attrs_d[..., 1:4],
                                         v_attrs_d[..., 4:])
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            voxelgrid_colors=colors_d,
            training=training
        )

        mesh = MeshExtractResult(vertices=vertices, faces=faces,
                                 vertex_attrs=colors, res=self.res)

        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]

        return mesh
