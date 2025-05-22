from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from ..basic import BasicTrainer
from ...representations import MeshExtractResult
from ...renderers import MeshRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, smooth_l1_loss, ssim, lpips
from ...utils.data_utils import recursive_to_device


class SLatVaeMeshDecoderTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE Mesh Decoder.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
    """
    
    def __init__(
        self,
        *args,
        depth_loss_type: str = 'l1',
        lambda_depth: int = 1,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_tsdf: float = 0.01,
        lambda_color: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_loss_type = depth_loss_type
        self.lambda_depth = lambda_depth
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_tsdf = lambda_tsdf
        self.lambda_color = lambda_color
        self.use_color = self.lambda_color > 0
        
        self._init_renderer()
        
    def _init_renderer(self):
        rendering_options = {"near" : 1,
                             "far" : 3}
        self.renderer = MeshRenderer(rendering_options, device=self.device)
        
    def _render_batch(self, reps: List[MeshExtractResult], extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                      return_types=['mask', 'normal', 'depth']) -> Dict[str, torch.Tensor]:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_types: vary in ['mask', 'normal', 'depth', 'normal_map', 'color']
            
        Returns: 
            a dict with
                reg_loss : [N] tensor of regularization losses
                mask : [N x 1 x H x W] tensor of rendered masks
                normal : [N x 3 x H x W] tensor of rendered normals
                depth : [N x 1 x H x W] tensor of rendered depths
        """
        ret = {k : [] for k in return_types}
        for i, rep in enumerate(reps):
            out_dict = self.renderer.render(rep, extrinsics[i], intrinsics[i], return_types=return_types)
            for k in out_dict:
                ret[k].append(out_dict[k][None] if k in ['mask', 'depth'] else out_dict[k])
        for k in ret:
            ret[k] = torch.stack(ret[k])
        return ret
    
    @staticmethod
    def _tsdf_reg_loss(rep: MeshExtractResult, depth_map: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        # Calculate tsdf
        with torch.no_grad():
            # Project points to camera and calculate pseudo-sdf as difference between gt depth and projected depth
            projected_pts, pts_depth = utils3d.torch.project_cv(extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v)
            projected_pts = (projected_pts - 0.5) * 2.0
            depth_map_res = depth_map.shape[1]
            gt_depth = torch.nn.functional.grid_sample(depth_map.reshape(1, 1, depth_map_res, depth_map_res), 
            projected_pts.reshape(1, 1, -1, 2), mode='bilinear', padding_mode='border', align_corners=True)
            pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
            # Truncate pseudo-sdf
            delta = 1 / rep.res * 3.0
            trunc_mask = pseudo_sdf > -delta
        
        # Loss
        gt_tsdf = pseudo_sdf[trunc_mask]
        tsdf = rep.tsdf_s.flatten()[trunc_mask]
        gt_tsdf = torch.clamp(gt_tsdf, -delta, delta)
        return torch.mean((tsdf - gt_tsdf) ** 2)
    
    def _calc_tsdf_loss(self, reps : list[MeshExtractResult], depth_maps, extrinsics, intrinsics) -> torch.Tensor:
        tsdf_loss = 0.0
        for i, rep in enumerate(reps):
            tsdf_loss += self._tsdf_reg_loss(rep, depth_maps[i], extrinsics[i], intrinsics[i])
        return tsdf_loss / len(reps)
    
    @torch.no_grad()
    def _flip_normal(self, normal: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Flip normal to align with camera.
        """
        normal = normal * 2.0 - 1.0
        R = torch.zeros_like(extrinsics)
        R[:, :3, :3] = extrinsics[:, :3, :3]
        R[:, 3, 3] = 1.0
        view_dir = utils3d.torch.unproject_cv(
            utils3d.torch.image_uv(*normal.shape[-2:], device=self.device).reshape(1, -1, 2),
            torch.ones(*normal.shape[-2:], device=self.device).reshape(1, -1),
            R, intrinsics
        ).reshape(-1, *normal.shape[-2:], 3).permute(0, 3, 1, 2)
        unflip = (normal * view_dir).sum(1, keepdim=True) < 0
        normal *= unflip * 2.0 - 1.0
        return (normal + 1.0) / 2.0
    
    def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
        """
        Combination of L1, SSIM, and LPIPS loss.
        """
        if gt.shape[1] != 3:
            assert gt.shape[-1] == 3
            gt = gt.permute(0, 3, 1, 2)
        if pred.shape[1] != 3:
            assert pred.shape[-1] == 3
            pred = pred.permute(0, 3, 1, 2)
        terms = {
            f"{name}_loss" : l1_loss(gt, pred),
            f"{name}_loss_ssim" : 1 - ssim(gt, pred),
            f"{name}_loss_lpips" : lpips(gt, pred)
        }
        terms[f"{name}_loss_perceptual"] = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim + terms[f"{name}_loss_lpips"] * self.lambda_lpips
        return terms
    
    def geometry_losses(
        self,
        reps: List[MeshExtractResult],
        mesh: List[Dict],
        normal_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        with torch.no_grad():
            gt_meshes = []
            for i in range(len(reps)):
                gt_mesh = MeshExtractResult(mesh[i]['vertices'].to(self.device), mesh[i]['faces'].to(self.device))
                gt_meshes.append(gt_mesh)
            target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=['mask', 'depth', 'normal'])
            target['normal'] = self._flip_normal(target['normal'], extrinsics, intrinsics)
                
        terms = edict(geo_loss = 0.0)
        if self.lambda_tsdf > 0:
            tsdf_loss = self._calc_tsdf_loss(reps, target['depth'], extrinsics, intrinsics)
            terms['tsdf_loss'] = tsdf_loss
            terms['geo_loss'] += tsdf_loss * self.lambda_tsdf
        
        return_types = ['mask', 'depth', 'normal', 'normal_map'] if self.use_color else ['mask', 'depth', 'normal']
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
        
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            for k, v in buffer.items():
                buffer[k] = v[success_mask]
            for k, v in target.items():
                target[k] = v[success_mask]
            
            terms['mask_loss'] = l1_loss(buffer['mask'], target['mask']) 
            if self.depth_loss_type == 'l1':
                terms['depth_loss'] = l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'])
            elif self.depth_loss_type == 'smooth_l1':
                terms['depth_loss'] = smooth_l1_loss(buffer['depth'] * target['mask'], target['depth'] * target['mask'], beta=1.0 / (2 * reps[0].res))
            else:
                raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
            terms.update(self._perceptual_loss(buffer['normal'] * target['mask'], target['normal'] * target['mask'], 'normal'))
            terms['geo_loss'] = terms['geo_loss'] + terms['mask_loss'] + terms['depth_loss'] * self.lambda_depth + terms['normal_loss_perceptual']
            if self.use_color and normal_map is not None:
                terms.update(self._perceptual_loss(normal_map[success_mask], buffer['normal_map'], 'normal_map'))
                terms['geo_loss'] = terms['geo_loss'] + terms['normal_map_loss_perceptual'] * self.lambda_color
                
        return terms
      
    def color_losses(self, reps, image, alpha, extrinsics, intrinsics):
        terms = edict(color_loss = torch.tensor(0.0, device=self.device))
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=['color'])
        success_mask = torch.tensor([rep.success for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            terms.update(self._perceptual_loss((image * alpha[:, None])[success_mask], buffer['color'][success_mask], 'color'))
            terms['color_loss'] = terms['color_loss'] + terms['color_loss_perceptual'] * self.lambda_color
        return terms
    
    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        mesh: List[Dict],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        normal_map: torch.Tensor = None,
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            latents: The [N x * x C] sparse latents
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            mesh: The list of dictionaries of meshes.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        reps = self.training_models['decoder'](latents)
        self.renderer.rendering_options.resolution = image.shape[-1]
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        terms['reg_loss'] = sum([rep.reg_loss for rep in reps]) / len(reps)
        terms['loss'] = terms['loss'] + terms['reg_loss']
        
        geo_terms = self.geometry_losses(reps, mesh, normal_map, extrinsics, intrinsics)
        terms.update(geo_terms)
        terms['loss'] = terms['loss'] + terms['geo_loss']
                
        if self.use_color:
            color_terms = self.color_losses(reps, image, alpha, extrinsics, intrinsics)
            terms.update(color_terms)
            terms['loss'] = terms['loss'] + terms['color_loss']
             
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        ret_dict = {}
        gt_images = []
        gt_normal_maps = []
        gt_meshes = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = recursive_to_device(data, 'cuda')
            gt_images.append(args['image'] * args['alpha'][:, None])
            if self.use_color and 'normal_map' in data:
                gt_normal_maps.append(args['normal_map'])
            gt_meshes.extend(args['mesh'])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            reps.extend(self.models['decoder'](args['latents']))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        if self.use_color and gt_normal_maps:
            gt_normal_maps = torch.cat(gt_normal_maps, dim=0)
            ret_dict.update({f'gt_normal_map': {'value': gt_normal_maps, 'type': 'image'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        gt_render_results = self._render_batch([
            MeshExtractResult(vertices=mesh['vertices'].to(self.device), faces=mesh['faces'].to(self.device))
            for mesh in gt_meshes
        ], exts, ints, return_types=['normal'])
        ret_dict.update({f'gt_normal': {'value': self._flip_normal(gt_render_results['normal'], exts, ints), 'type': 'image'}})
        return_types = ['normal']
        if self.use_color:
            return_types.append('color')
            if 'normal_map' in data:
                return_types.append('normal_map')
        render_results = self._render_batch(reps, exts, ints, return_types=return_types)
        ret_dict.update({f'rec_normal': {'value': render_results['normal'], 'type': 'image'}})
        if 'color' in return_types:
            ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})
        if 'normal_map' in return_types:
            ret_dict.update({f'rec_normal_map': {'value': render_results['normal_map'], 'type': 'image'}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        multiview_normals = []
        multiview_normal_maps = []
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)
            multiview_normals.append(render_results['normal'])
            if 'color' in return_types:
                miltiview_images.append(render_results['color'])
            if 'normal_map' in return_types:
                multiview_normal_maps.append(render_results['normal_map'])

        ## Concatenate views
        multiview_normals = torch.cat([
            torch.cat(multiview_normals[:2], dim=-2),
            torch.cat(multiview_normals[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'multiview_normal': {'value': multiview_normals, 'type': 'image'}})
        if 'color' in return_types:
            miltiview_images = torch.cat([
                torch.cat(miltiview_images[:2], dim=-2),
                torch.cat(miltiview_images[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_image': {'value': miltiview_images, 'type': 'image'}})
        if 'normal_map' in return_types:
            multiview_normal_maps = torch.cat([
                torch.cat(multiview_normal_maps[:2], dim=-2),
                torch.cat(multiview_normal_maps[2:], dim=-2),
            ], dim=-1)
            ret_dict.update({f'multiview_normal_map': {'value': multiview_normal_maps, 'type': 'image'}})
                            
        return ret_dict
