import gsplat as gs
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


class GSplatRenderer:
    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False,
            "use_mip_gaussian": True
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None

    def render(
            self,
            gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
    ) -> edict:

        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]

        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"],
                dtype=torch.float32,
                device="cuda"
            )

        height = resolution * ssaa
        width = resolution * ssaa

        # Set up background color
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(
                self.rendering_options["bg_color"],
                dtype=torch.float32,
                device="cuda"
            )

        Ks_scaled = intrinsics.clone()
        Ks_scaled[0, 0] *= width
        Ks_scaled[1, 1] *= height
        Ks_scaled[0, 2] *= width
        Ks_scaled[1, 2] *= height
        Ks_scaled = Ks_scaled.unsqueeze(0)

        near_plane = 0.01
        far_plane = 1000.0

        # Rasterize with gsplat
        render_colors, render_alphas, meta = gs.rasterization(
            means=gaussian.get_xyz,
            quats=F.normalize(gaussian.get_rotation, dim=-1),
            scales=gaussian.get_scaling / intrinsics[0, 0],
            opacities=gaussian.get_opacity.squeeze(-1),
            colors=colors_overwrite.unsqueeze(0) if colors_overwrite is not None else torch.sigmoid(
                gaussian.get_features.squeeze(1)).unsqueeze(0),
            viewmats=extrinsics.unsqueeze(0),
            Ks=Ks_scaled,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=3.0,
            eps2d=0.3,
            render_mode="RGB",
            backgrounds=self.bg_color.unsqueeze(0),
            camera_model="pinhole"
        )

        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)

        # Apply supersampling if needed
        if ssaa > 1:
            rendered_image = F.interpolate(
                rendered_image[None],
                size=(resolution, resolution),
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).squeeze()

        return edict({'color': rendered_image})
