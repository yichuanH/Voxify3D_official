import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo

from . import grid
from .dvgo import Raw2Alpha, Alphas2Weights
from .dmpigo import create_full_step_id

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
ub360_utils_cuda = load(
        name='ub360_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/ub360_utils.cpp', 'cuda/ub360_utils_kernel.cu']],
        verbose=True)


'''Model'''
class DirectContractedVoxGO(nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_world_size=None,
                 fast_color_thres=0, bg_len=0.2,
                 contracted_norm='inf',
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectContractedVoxGO, self).__init__()
        # xyz_min/max are the boundary that separates fg and bg scene
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectContractedVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1,-1,-1]) - bg_len)
        self.register_buffer('xyz_max', torch.Tensor([1,1,1]) + bg_len)
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.bg_len = bg_len
        self.contracted_norm = contracted_norm

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dcvgo: set density bias shift to', self.act_shift)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dcvgo: feature voxel grid', self.k0)
            print('dcvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dcvgo: voxel_size      ', self.voxel_size)
        print('dcvgo: world_size      ', self.world_size)
        print('dcvgo: voxel_size_base ', self.voxel_size_base)
        print('dcvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'contracted_norm': self.contracted_norm,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dcvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dcvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dcvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dcvgo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, inner_mask, t = self.sample_ray(
                        ori_rays_o=rays_o.to(device), ori_rays_d=rays_d.to(device),
                        **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        eps_time = time.time() - eps_time
        print(f'dcvgo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius
        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        N_inner = int(2 / (2+2*self.bg_len) * self.world_len / stepsize) + 1
        N_outer = N_inner
        b_inner = torch.linspace(0, 2, N_inner+1)
        b_outer = 2 / torch.linspace(1, 1/128, N_outer+1)
        t = torch.cat([
            (b_inner[1:] + b_inner[:-1]) * 0.5,
            (b_outer[1:] + b_outer[:-1]) * 0.5,
        ])
        ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        inner_mask = (norm<=1)
        ray_pts = torch.where(
            inner_mask,
            ray_pts,
            ray_pts / norm * ((1+self.bg_len) - self.bg_len/norm)
        )
        return ray_pts, inner_mask.squeeze(-1), t

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, is_train=False, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            print(f'dcvgo: update fast_color_thres {self.fast_color_thres} => {self._fast_color_thres[global_step]}')
            self.fast_color_thres = self._fast_color_thres[global_step]

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, inner_mask, t = self.sample_ray(
                ori_rays_o=rays_o, ori_rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        dist_thres = (2+2*self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[None].repeat(N,1)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask.flatten()]

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            weights = weights[mask]

        # query for color
        k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and is_train:
            rgb_marched += (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        wsum_mid = segment_coo(
                src=weights[inner_mask],
                index=ray_id[inner_mask],
                out=torch.zeros([N]),
                reduce='sum')
        s = 1 - 1/(1+t)  # [0, inf] => [0, 1]
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'wsum_mid': wsum_mid,
            'rgb_marched': rgb_marched,
            'raw_density': density,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'step_id': step_id,
            'n_max': n_max,
            't': t,
            's': s,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


class DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, s, n_max, ray_id):
        n_rays = ray_id.max()+1
        interval = 1/n_max
        w_prefix, w_total, ws_prefix, ws_total = ub360_utils_cuda.segment_cumsum(w, s, ray_id)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (s * w_prefix - ws_prefix)
        ctx.save_for_backward(w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id)
        ctx.interval = interval
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id = ctx.saved_tensors
        interval = ctx.interval
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total[ray_id] - (w_prefix + w)
        ws_suffix = ws_total[ray_id] - (ws_prefix + w*s)
        grad_bi = 2 * (s * (w_prefix - w_suffix) + (ws_suffix - ws_prefix))
        grad = grad_back * (grad_bi + grad_uni)
        return grad, None, None, None





'''
Stochastic Model
'''
class DirectContractedVoxGO_Sto(nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_world_size=None,
                 fast_color_thres=0, bg_len=0.2,
                 uncertainty_mask=False,
                 density_std_init=1,
                 contracted_norm='inf',
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 K_samples=12,
                 **kwargs):
        super(DirectContractedVoxGO_Sto, self).__init__()
        # xyz_min/max are the boundary that separates fg and bg scene
        xyz_min = torch.Tensor(xyz_min)
        xyz_max = torch.Tensor(xyz_max)
        assert len(((xyz_max - xyz_min) * 100000).long().unique()), 'scene bbox must be a cube in DirectContractedVoxGO'
        self.register_buffer('scene_center', (xyz_min + xyz_max) * 0.5)
        self.register_buffer('scene_radius', (xyz_max - xyz_min) * 0.5)
        self.register_buffer('xyz_min', torch.Tensor([-1,-1,-1]) - bg_len)
        self.register_buffer('xyz_max', torch.Tensor([1,1,1]) + bg_len)
        if isinstance(fast_color_thres, dict):
            self._fast_color_thres = fast_color_thres
            self.fast_color_thres = fast_color_thres[0]
        else:
            self._fast_color_thres = None
            self.fast_color_thres = fast_color_thres
        self.bg_len = bg_len
        self.contracted_norm = contracted_norm
        self.uncertainty_mask = uncertainty_mask
        self.density_std_init = density_std_init
        self.K_samples = K_samples

        print('bg_len',bg_len)
        print('K_samples',K_samples)
        print('uncertainty_mask',uncertainty_mask)
        print('density_std_init',density_std_init)

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dcvgo: set density bias shift to', self.act_shift)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        
        # init density std voxel grid
        self.density_std = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        with torch.no_grad():
            self.density_std.grid.data += density_std_init
        
        # construct an uncertain voxel grid with initialized high uncertainty of value = 1
        self.uncertainty_grid = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        with torch.no_grad():
            self.uncertainty_grid.grid.data += 1

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            dim0 += self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 6)
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dcvgo: feature voxel grid', self.k0)
            print('dcvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
            path=None, mask=mask,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        

        self.sample_size = self.K_samples
        self.sample_den = torch.empty([1,self.sample_size]).normal_()
        self.sample_rgb = torch.empty([1,self.sample_size,3]).normal_()

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.world_len = self.world_size[0].item()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dcvgo: voxel_size      ', self.voxel_size)
        print('dcvgo: world_size      ', self.world_size)
        print('dcvgo: voxel_size_base ', self.voxel_size_base)
        print('dcvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'contracted_norm': self.contracted_norm,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'density_std_init': self.density_std_init,
            'bg_len': self.bg_len,
            'uncertainty_mask': self.uncertainty_mask,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dcvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dcvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.density_std.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        self.uncertainty_grid.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)

            cache_grid_density = self.density(self_grid_xyz) # (*world_size)
            cache_grid_density_std = self.density_std(self_grid_xyz) # (*world_size)
            print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
            print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
            den = cache_grid_density # TODO: consider density_std too

            # den_std = F.softplus(cache_grid_density_std) + 1e-03
            # eps_den = torch.Tensor([3.])
            # den = eps_den.mul(den_std).add_(den)

            cache_grid_alpha = self.activate_density(den)[None,None]
            self_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            ori_p = self.mask_cache.mask.float().mean().item()
            print(f'dcvgo: mask_cache after scale {ori_p:.4f}')    

        print('dcvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)

        cache_grid_density = self.density(cache_grid_xyz) # (*world_size)
        cache_grid_density_std = self.density_std(cache_grid_xyz) # (*world_size)
        print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
        print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
        den = cache_grid_density # TODO: consider density_std too

        # den_std = F.softplus(cache_grid_density_std) + 1e-03
        # eps_den = torch.Tensor([3.])
        # den = eps_den.mul(den_std).add_(den)

        cache_grid_alpha = self.activate_density(den)[None,None]
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dcvgo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, inner_mask, t = self.sample_ray(
                        ori_rays_o=rays_o.to(device), ori_rays_d=rays_d.to(device),
                        **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        eps_time = time.time() - eps_time
        print(f'dcvgo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)
        self.density_std.total_variation_add_grad(w, w, w, dense_mode)
        
    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)
    
    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None,None] <= near_clip] = -100
        self.density_std.grid[nearest_dist[None,None] <= near_clip] = -100

    def sample_ray(self, ori_rays_o, ori_rays_d, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = (ori_rays_o - self.scene_center) / self.scene_radius

        rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
        N_inner = int(2 / (2+2*self.bg_len) * self.world_len / stepsize) + 1

        N_outer = N_inner 
        b_inner = torch.linspace(0, 2, N_inner+1)
        b_outer = 2 / torch.linspace(1, 1/128, N_outer+1)
        t = torch.cat([
            (b_inner[1:] + b_inner[:-1]) * 0.5,
            (b_outer[1:] + b_outer[:-1]) * 0.5,
        ])
        ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]
        if self.contracted_norm == 'inf':
            norm = ray_pts.abs().amax(dim=-1, keepdim=True)
        elif self.contracted_norm == 'l2':
            norm = ray_pts.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        inner_mask = (norm<=1)
        ray_pts = torch.where(
            inner_mask,
            ray_pts,
            ray_pts / norm * ((1+self.bg_len) - self.bg_len/norm)
        )
        return ray_pts, inner_mask.squeeze(-1), t

    @torch.no_grad()
    def set_zero_unseen_foreground(self, rays_o, rays_d, viewdirs, N_rand=0, global_step=-1, N_iters=-1, **render_kwargs):

        # sample points on rays
        ray_pts, inner_mask, _ = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        dist_thres = (2+2*self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        mask_unc = (self.uncertainty_grid(ray_pts)>0)
        self.uncertainty_grid.set_zero_at(ray_pts[mask_unc], self.world_size)
        
        return
    
    @torch.no_grad()
    def update_uncertainty(self, rays_o, rays_d, viewdirs, N_rand=0, global_step=-1, N_iters=-1, **render_kwargs):

        N = len(rays_o)

        # sample points on rays
        ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
        ray_pts_ori = ray_pts.clone()
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        step_id = step_id.reshape([N,-1])
        # set a boundary for foreground
        step_id = torch.where(
            inner_mask,
            0,
            step_id
        )

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        dist_thres = (2+2*self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[None].repeat(N,1)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask]

        # set low uncertainty for voxels when rays hit scene geometry outside the non-contracted space (foreground)
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        weights, alphainv_last, T, i_end = Alphas2Weights.apply(alpha, ray_id, N)
        i_end = torch.clamp(i_end.int(), max=step_id.shape[0]-1)
        step_id_end = torch.index_select(step_id.reshape(-1), dim=0, index=i_end)
        mask = (step_id_end > 0)
        ray_pts_ori = ray_pts_ori[mask].view(-1,3)
        mask_unc = (self.uncertainty_grid(ray_pts_ori)>0)
        self.uncertainty_grid.set_zero_at(ray_pts_ori[mask_unc], self.world_size)

        # set low uncertainty for the surface voxels 
        mask = (T > 0.5)
        empty_ray_pts = ray_pts[mask]
        mask_unc = (self.uncertainty_grid(empty_ray_pts)>0)
        self.uncertainty_grid.set_zero_at(empty_ray_pts[mask_unc], self.world_size)
        
        return
    
    def forward(self, rays_o, rays_d, viewdirs, N_rand=0, global_step=-1, N_iters=-1, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only support point queries in [N, 3] format'
        if isinstance(self._fast_color_thres, dict) and global_step in self._fast_color_thres:
            print(f'dcvgo: update fast_color_thres {self.fast_color_thres} => {self._fast_color_thres[global_step]}')
            self.fast_color_thres = self._fast_color_thres[global_step]

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
        n_max = len(t)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # set low uncertainty for voxels near clip 
        if render_kwargs.get('is_test', False):
            # render_kwargs['near_clip'] = 0.6
            near_clip = render_kwargs['near_clip'] * 1.5
            near_mask = (t[None].repeat(N,1) < near_clip)
            ray_pts_near_clip = ray_pts[near_mask]
            mask_unc = (self.uncertainty_grid(ray_pts_near_clip)>0)
            ray_pts_near_clip = ray_pts_near_clip[mask_unc]
            self.uncertainty_grid.set_zero_at(ray_pts_near_clip, self.world_size)
            self.uncertainty_grid.update_indices(self.world_size)
        
        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        dist_thres = (2+2*self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
        dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[None].repeat(N,1)[mask]
        ray_id = ray_id[mask.flatten()]
        step_id = step_id[mask.flatten()]

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]
        t = t[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

        # query for alpha w/ post-activation
        den_mean = self.density(ray_pts)
        den_std = F.softplus(self.density_std(ray_pts)) + 1e-3
        loss_entropy_den = - den_std.log()
        if render_kwargs.get('is_train', False):
            eps_den = torch.empty([1,self.K_samples]).normal_()
        else:
            eps_den = self.sample_den
        raw_density_k = eps_den.mul(den_std[:,None]).add_(den_mean[:,None]) # (BN,K)
        alpha_k = self.activate_density(raw_density_k, interval)
        if self.fast_color_thres > 0:
            alpha = alpha_k.mean(-1)
            mask = (alpha > self.fast_color_thres)
            alpha_std = alpha_k.std(-1)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            raw_density_k = raw_density_k[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            alpha_k = alpha_k[mask]
            loss_entropy_den = loss_entropy_den[mask]
            den_std = den_std[mask]
            alpha_std = alpha_std[mask]

        # new compute accumulated transmittance
        shape = alpha_k.shape
        weights_k, alphainv_last_k, T_k, i_end = Alphas2Weights_k.apply(alpha_k, ray_id, N)
        alphainv_last_k = alphainv_last_k.reshape([N,self.K_samples])
        if self.fast_color_thres > 0:
            weights = weights_k.mean(-1)
            mask = (weights > self.fast_color_thres)
            weights_std = weights_k.std(-1)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            raw_density_k = raw_density_k[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
            weights_k = weights_k[mask]
            T_k = T_k[mask]
            loss_entropy_den = loss_entropy_den[mask]
            den_std = den_std[mask]
            alpha_std = alpha_std[mask]
            weights_std = weights_std[mask]

        # query for color
        k0 = self.k0(ray_pts)
        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0, viewdirs_emb], -1)
            raw_rgb = self.rgbnet(rgb_feat)
            
            rgb_mean = raw_rgb[:,:3]
            rgb_std = F.softplus(raw_rgb[:,3:]) + 1e-3
            loss_entropy_rgb = - rgb_std.log().mean(-1)
            if render_kwargs.get('is_train', False):
                eps_rgb = torch.empty([1,self.K_samples, 3]).normal_()
            else:
                eps_rgb = self.sample_rgb
            raw_rgb_k = eps_rgb.mul(rgb_std[:,None]).add_(rgb_mean[:,None]) # (BN,K,3)
            rgb_k = torch.sigmoid(raw_rgb_k)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights_k[...,None] * rgb_k),
                index=ray_id,
                out=torch.zeros([N, self.K_samples, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and render_kwargs.get('is_train', False):
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * render_kwargs['bg'])

        s = 1 - 1/(1+t)
        ret_dict.update({
            'rgb_marched': rgb_marched,
            'alphainv_last': alphainv_last_k.mean(-1),
            'alphainv_last_k': alphainv_last_k,
            'n_max': torch.Tensor([n_max]),
            'ray_pts': ray_pts,
            'weights_k': weights_k,
            'weights': weights,
            'raw_rgb_k': rgb_k,
            'raw_density_k': raw_density_k,
            'ray_id': ray_id,
            't': t,
            's': s,
            'loss_entropy_den': loss_entropy_den,
            'loss_entropy_rgb': loss_entropy_rgb,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth_k = segment_coo(
                        src=(weights_k * s.unsqueeze(-1)),
                        index=ray_id,
                        out=torch.zeros([N, self.K_samples]),
                        reduce='sum')
                # acc_map = segment_coo(
                #         src=weights_k,
                #         index=ray_id,
                #         out=torch.zeros([N, self.K_samples]),
                #         reduce='sum')
                # disp_k = 1./torch.max(1e-10 * torch.ones_like(depth_k), depth_k / (acc_map + 1e-10))
                # disp_k = 1./(depth_k + 1e-10)
                # disp = 1 - depth / np.max(depth)
                ret_dict.update({'render_depth': depth_k})

                # baseline: compute epistemic uncertainty term from the paper: 'Density-aware NeRF Ensembles: Quantifying Predictive Uncertainty in Neural Radiance Fields'
                U_epi = segment_coo(
                    src=weights,
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum')
                ret_dict.update({'U_epi': (1-U_epi)**2})

        if render_kwargs.get('compute_VH', False): 

            # sample points on rays
            ray_pts, inner_mask, t = self.sample_ray(ori_rays_o=rays_o, ori_rays_d=rays_d, **render_kwargs)
            n_max = len(t)
            interval = render_kwargs['stepsize'] * self.voxel_size_ratio
            ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

            # skip oversampled points outside scene bbox
            mask = inner_mask.clone()
            dist_thres = (2+2*self.bg_len) / self.world_len * render_kwargs['stepsize'] * 0.95
            dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
            mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[None].repeat(N,1)[mask]
            ray_id = ray_id[mask.flatten()]
            step_id = step_id[mask.flatten()]

            # skip known free space
            mask = self.mask_cache(ray_pts)
            mask |= (self.uncertainty_grid(ray_pts)>0)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

            # query for alpha w/ post-activation
            density = self.density(ray_pts)
            alpha = self.activate_density(density, interval)
            if self.fast_color_thres > 0:
                mask = (alpha > self.fast_color_thres)
                mask |= (self.uncertainty_grid(ray_pts)>0)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                alpha = alpha[mask]

            weights, alphainv_last, T, i_end = Alphas2Weights.apply(alpha, ray_id, N)
            T = T.pow(4)
            VH = self.uncertainty_grid(ray_pts)
            # VH = torch.zeros_like(VH).cuda()
            U_VH = segment_coo(
                src=(T * VH),
                index=ray_id,
                out=torch.zeros([N]),
                reduce='sum')
            ret_dict.update({'U_VH': U_VH})

        return ret_dict


class DistortionLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, s, n_max, ray_id):
        n_rays = ray_id.max()+1
        interval = 1/n_max
        w_prefix, w_total, ws_prefix, ws_total = ub360_utils_cuda.segment_cumsum(w, s, ray_id)
        loss_uni = (1/3) * interval * w.pow(2)
        loss_bi = 2 * w * (s * w_prefix - ws_prefix)
        ctx.save_for_backward(w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id)
        ctx.interval = interval
        return (loss_bi.sum() + loss_uni.sum()) / n_rays

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        w, s, w_prefix, w_total, ws_prefix, ws_total, ray_id = ctx.saved_tensors
        interval = ctx.interval
        grad_uni = (1/3) * interval * 2 * w
        w_suffix = w_total[ray_id] - (w_prefix + w)
        ws_suffix = ws_total[ray_id] - (ws_prefix + w*s)
        grad_bi = 2 * (s * (w_prefix - w_suffix) + (ws_suffix - ws_prefix))
        grad = grad_back * (grad_bi + grad_uni)
        return grad, None, None, None

distortion_loss = DistortionLoss.apply

