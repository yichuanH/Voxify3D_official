import os
import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


grid_mode = "nearest" # for uncertainty training
#grid_mode = "bilinear"
grid_mode_2 = "bilinear"
#grid_mode_2 = 'nearest'  ### try 都改nearest看看，剛剛是bilinear

# -------- return grid class ------- #
def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    elif type == 'LogitGrid':
        #breakpoint()
        return LogitGrid(**kwargs)
    else:
        raise NotImplementedError


'''
Dense 3D grid
'''
class LogitGrid(nn.Module):
    def __init__(self, palette, world_size, xyz_min, xyz_max, tau=0.3, hard=True, color_num=8, **kwargs):  ### tau 1.0 -> 0.1
        """
        palette: (C, 3) RGB palette tensor, values in [0, 1]
        world_size: (D, H, W)
        """
        super().__init__()
        self.palette = palette  # 固定的調色盤，不需要訓練
        self.C = color_num
        self.channels = color_num
        self.world_size = world_size
        self.tau = tau
        self.hard = hard

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        # 建立 Logit Grid：每格是一組 logits（長度 C）
        self.grid = nn.Parameter(torch.randn([1, self.C, *world_size]) * 0.01)
        #breakpoint()
        
    def forward(self, xyz, global_step):
        """
        xyz: global coordinates, shape (N, M, 3) or (B, ..., 3)
        return: RGB values (N, M, 3) via Gumbel-Softmax from palette
        """
        #print(">>> LogitGrid.forward called!")
        shape = xyz.shape[:-1]     # e.g., (N, M) 
        xyz = xyz.reshape(1, 1, 1, -1, 3)  # (1, 1, 1, N*M, 3)
        
        # normalize to [-1, 1] and flip
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(-1) * 2 - 1  # (1, 1, 1, N*M, 3)
        
        # 1. grid logit 查值 + 加 gumbel noise
        # Grid sample: input (1, C, D, H, W), grid (1, 1, 1, N*M, 3)
        logits = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)  # (1, C, 1, 1, N*M)  
        ### Gumbel(logit_i) = logit_i + gumbel_noise_i
        
        #print("logits after grid_sample:", logits)
        logits = logits.view(self.C, -1).T.view(*shape, self.C)  # (N, M, C)
        #print("self.palette = ", self.palette)

        logits.retain_grad()
        self._last_logits = logits 
        
        ### dynamic tau
        # 🔧 設定動態 tau 分段調整
        if global_step is not None:
            # 前 0–1000 iter（bt+bk+tp）：tau = [0.8, 0.6, 0.8]
            if global_step < 1000:
                tau_now = 1.0
            elif global_step < 3000:
                tau_now = 0.8
            elif global_step < 4000:
                tau_now = 0.3  ###0.7
            elif global_step < 5000:
                tau_now = 0.6
            elif global_step <= 6000:
                tau_now = 0.3
            elif global_step <= 7000:
                tau_now = 0.1
            
        else:
            tau_now = self.tau  # default fallback if step not provided

        if global_step <5000:
            probs = F.gumbel_softmax(logits, tau=tau_now, hard=False)
            rgb = torch.matmul(probs, self.palette.to(probs.device))  # 確保同一個 device
        else:
            probs = F.gumbel_softmax(logits, tau=tau_now, hard=True)  #True
            rgb = torch.matmul(probs, self.palette.to(probs.device))  # 確保同一個 device
                
        #breakpoint()
        return rgb  

    
    def get_nearest(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]     # 10000, 174  
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode=grid_mode, align_corners=True)  #
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        #print(f"xyz.shape before reshaping in get_nearest: {xyz.shape}")
        return out


    ### ----- Voxel Interpolation for Volume Rendeing ------- #
    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode=grid_mode_2, align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_logit_grid(self):
        grid = self.grid
        print("DenseGrid Voxel Representation:")
        print(grid.shape)  # [1, 1, 105, 104, 92] 
        return grid  

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'
    
    @torch.no_grad()
    def reset(self, new_world_size, datatype):
        self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))

        if datatype == 'unbounded':
            # unbounded scenes involves ray parameterization for the unbounded background and thus need more complex initialization!
            # firstly, init high value in the uncertainty field for the non-contracted space (foreground space, [-1,1]), involving points O,E,S in the paper
            normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
            normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
            fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
            fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
            self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]] += 1
            # Secondly, init high uncertainty value for only boundary of the contracted space (background space), involving points P in the paper  
            pad = 3
            self.grid.data[:,:,:pad] += 1
            self.grid.data[:,:,-pad:] += 1
            self.grid.data[:,:,:,:pad] += 1
            self.grid.data[:,:,:,-pad:] += 1
            self.grid.data[:,:,:,:,:pad] += 1
            self.grid.data[:,:,:,:,-pad:] += 1

        if datatype == 'bounded':
            self.grid.data += 1

        print('reset uncertainty grid:',(self.grid.data==0).sum()/torch.prod(new_world_size))

    def Normalize(self, xyz):
        return (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)


    ''' Vector-Matrix decomposited grid
    See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
    '''
    # Used for update the uncertainty grid with the sampled points
    @torch.no_grad()
    def set_zero_at(self, xyz, new_world_size):
        # Scale xyz from range (xyz_min, xyz_max) to (0,1)
        normalized_xyz = self.Normalize(xyz)

        # Convert normalized values to grid indices
        base_indices = torch.unique(torch.round(normalized_xyz * (new_world_size - 1)).long(), dim=0)

        if self.indices_to_update == None:
            self.indices_to_update = base_indices
        else:
            self.indices_to_update = torch.unique(torch.cat([self.indices_to_update, base_indices],0), dim=0)
    


    @torch.no_grad()
    def update_indices(self, new_world_size):
        '''
        Given a ray point, set the nearest vertex to be of zero uncertainty 
        '''
        if self.indices_to_update is None:
            return
        # # Broadcast to get all the indices for the 2x2x2 neighbors for each point
        # offsets = torch.tensor([
        #     [0, 0, 0],
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [0, 1, 1],
        #     [1, 0, 0],
        #     [1, 0, 1],
        #     [1, 1, 0],
        #     [1, 1, 1]
        # ])
        # all_indices = self.indices_to_update[:, None] + offsets
        # unique_indices = torch.unique(all_indices.reshape(-1,3), dim=0)

        # Ensure the indices are within the grid boundaries
        unique_indices = torch.clamp(self.indices_to_update, max=new_world_size-1)
        # print('unique_indices updated: ',unique_indices.shape[0]/torch.prod(new_world_size))

        # Update the values at these indices to 0
        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] = 0  
        # print('uncertainty_grid updated: ',(self.grid.data==0).sum()/torch.prod(new_world_size))

        # Ensure the indices on the grid boundaries are zero
        self.grid.data[:,:,:1] = 0
        self.grid.data[:,:,-1:] = 0
        self.grid.data[:,:,:,:1] = 0
        self.grid.data[:,:,:,-1:] = 0
        self.grid.data[:,:,:,:,:1] = 0
        self.grid.data[:,:,:,:,-1:] = 0

        self.indices_to_update = None

    @torch.no_grad()
    def update_indices_unseen_foreground(self, new_world_size):
        if self.indices_to_update is None:
            return
        
        unique_indices = torch.clamp(self.indices_to_update,  max=new_world_size-1)

        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] += 1 
        
        
        normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
        normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
        fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
        fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
        
        temp = self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]]
        temp[temp==1] = 0

        self.indices_to_update = None

        return 
    
    @torch.no_grad()
    def maskout(self, xyz_fine_min, xyz_fine_max):

        shape = torch.tensor([*self.grid.data.shape[2:]])-1
        
        def normalize(xyz): return ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * shape).int()
        
        xyz_fine_min_norm = normalize(xyz_fine_min)
        xyz_fine_max_norm = normalize(xyz_fine_max)

        self.grid.data[:,:,:xyz_fine_min_norm[0]] = 0 
        self.grid.data[:,:,:,:xyz_fine_min_norm[1],:] = 0 
        self.grid.data[:,:,:,:,:xyz_fine_min_norm[2]] = 0 
        self.grid.data[:,:,xyz_fine_max_norm[0]:] = 0 
        self.grid.data[:,:,:,xyz_fine_max_norm[1]:] = 0
        self.grid.data[:,:,:,:,xyz_fine_max_norm[2]:] = 0  


class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size # 256 
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))  # 3D Voxel Space

        # defined to save the indices for updating uncertainty_grid
        self.indices_to_update = None
    
    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        # xyz [10000, 340, 3]  #### 10000 rays, 340 sp points, 3 xyz
        #breakpoint()

        shape = xyz.shape[:-1]     # 10000, 340
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode=grid_mode_2, align_corners=True)  #
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)

        #print("forward")
        #print(f"xyz.shape before reshaping in forward: {xyz.shape}")
        #breakpoint()
        
        return out ###10000, 340


    def get_nearest(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]     # 10000, 174  
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode=grid_mode, align_corners=True)  #
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        #print(f"xyz.shape before reshaping in get_nearest: {xyz.shape}")
        return out


    ### ----- Voxel Interpolation for Volume Rendeing ------- #
    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode=grid_mode_2, align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        grid = self.grid
        print("DenseGrid Voxel Representation:")
        print(grid.shape)  # [1, 1, 105, 104, 92] 
        return grid  

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'
    
    @torch.no_grad()
    def reset(self, new_world_size, datatype):
        self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))

        if datatype == 'unbounded':
            # unbounded scenes involves ray parameterization for the unbounded background and thus need more complex initialization!
            # firstly, init high value in the uncertainty field for the non-contracted space (foreground space, [-1,1]), involving points O,E,S in the paper
            normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
            normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
            fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
            fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
            self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]] += 1
            # Secondly, init high uncertainty value for only boundary of the contracted space (background space), involving points P in the paper  
            pad = 3
            self.grid.data[:,:,:pad] += 1
            self.grid.data[:,:,-pad:] += 1
            self.grid.data[:,:,:,:pad] += 1
            self.grid.data[:,:,:,-pad:] += 1
            self.grid.data[:,:,:,:,:pad] += 1
            self.grid.data[:,:,:,:,-pad:] += 1

        if datatype == 'bounded':
            self.grid.data += 1

        print('reset uncertainty grid:',(self.grid.data==0).sum()/torch.prod(new_world_size))

    def Normalize(self, xyz):
        return (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)


    ''' Vector-Matrix decomposited grid
    See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
    '''
    # Used for update the uncertainty grid with the sampled points
    @torch.no_grad()
    def set_zero_at(self, xyz, new_world_size):
        # Scale xyz from range (xyz_min, xyz_max) to (0,1)
        normalized_xyz = self.Normalize(xyz)

        # Convert normalized values to grid indices
        base_indices = torch.unique(torch.round(normalized_xyz * (new_world_size - 1)).long(), dim=0)

        if self.indices_to_update == None:
            self.indices_to_update = base_indices
        else:
            self.indices_to_update = torch.unique(torch.cat([self.indices_to_update, base_indices],0), dim=0)
    


    @torch.no_grad()
    def update_indices(self, new_world_size):
        '''
        Given a ray point, set the nearest vertex to be of zero uncertainty 
        '''
        if self.indices_to_update is None:
            return
        # # Broadcast to get all the indices for the 2x2x2 neighbors for each point
        # offsets = torch.tensor([
        #     [0, 0, 0],
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [0, 1, 1],
        #     [1, 0, 0],
        #     [1, 0, 1],
        #     [1, 1, 0],
        #     [1, 1, 1]
        # ])
        # all_indices = self.indices_to_update[:, None] + offsets
        # unique_indices = torch.unique(all_indices.reshape(-1,3), dim=0)

        # Ensure the indices are within the grid boundaries
        unique_indices = torch.clamp(self.indices_to_update, max=new_world_size-1)
        # print('unique_indices updated: ',unique_indices.shape[0]/torch.prod(new_world_size))

        # Update the values at these indices to 0
        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] = 0  
        # print('uncertainty_grid updated: ',(self.grid.data==0).sum()/torch.prod(new_world_size))

        # Ensure the indices on the grid boundaries are zero
        self.grid.data[:,:,:1] = 0
        self.grid.data[:,:,-1:] = 0
        self.grid.data[:,:,:,:1] = 0
        self.grid.data[:,:,:,-1:] = 0
        self.grid.data[:,:,:,:,:1] = 0
        self.grid.data[:,:,:,:,-1:] = 0

        self.indices_to_update = None

    @torch.no_grad()
    def update_indices_unseen_foreground(self, new_world_size):
        if self.indices_to_update is None:
            return
        
        unique_indices = torch.clamp(self.indices_to_update,  max=new_world_size-1)

        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] += 1 
        
        
        normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
        normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
        fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
        fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
        
        temp = self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]]
        temp[temp==1] = 0

        self.indices_to_update = None

        return 
    
    @torch.no_grad()
    def maskout(self, xyz_fine_min, xyz_fine_max):

        shape = torch.tensor([*self.grid.data.shape[2:]])-1
        
        def normalize(xyz): return ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * shape).int()
        
        xyz_fine_min_norm = normalize(xyz_fine_min)
        xyz_fine_max_norm = normalize(xyz_fine_max)

        self.grid.data[:,:,:xyz_fine_min_norm[0]] = 0 
        self.grid.data[:,:,:,:xyz_fine_min_norm[1],:] = 0 
        self.grid.data[:,:,:,:,:xyz_fine_min_norm[2]] = 0 
        self.grid.data[:,:,xyz_fine_max_norm[0]:] = 0 
        self.grid.data[:,:,:,xyz_fine_max_norm[1]:] = 0
        self.grid.data[:,:,:,:,xyz_fine_max_norm[2]:] = 0        

''' Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
'''

class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)
        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R+R+Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[...,[0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape,self.channels)
        else:
            out = compute_tensorf_val(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        
        #breakpoint()
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode=grid_mode_2, align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode=grid_mode_2, align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode=grid_mode_2, align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X,1], mode=grid_mode_2, align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y,1], mode=grid_mode_2, align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z,1], mode=grid_mode_2, align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.x_vec[:,:,1:], self.x_vec[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.y_vec[:,:,1:], self.y_vec[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.z_vec[:,:,1:], self.z_vec[:,:,:-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([
                torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0,:,:,0]),
                torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0,:,:,0]),
                torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0,:,:,0]),
            ])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0,:,:,0]) + \
                   torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0,:,:,0]) + \
                   torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0,:,:,0])
            grid = grid[None,None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config["n_comp"]}'

def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = torch.cat([
        xy_feat * z_feat,
        xz_feat * y_feat,
        yz_feat * x_feat,
    ], dim=-1)
    feat = torch.mm(feat, f_vec)
    return feat

def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = F.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    xz_feat = F.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    yz_feat = F.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    x_feat = F.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    y_feat = F.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    z_feat = F.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode=grid_mode_2, align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat

''' 
Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

