import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

from lib import dvgo_6views_old
import mmcv
import imageio
import numpy as np
import mmengine

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dcvgo, dmpigo
from lib import dvgo_6views_old as dvgo_ortho
from lib.load_data import load_data
from lib.rgb2logit import convert_densegrid_to_logitgrid, random_densegrid_to_logitgrid

from torch_efficient_distloss import flatten_eff_distloss
from torch import tensor
import matplotlib.pyplot as plt
from CLIP.clip_loss2 import CLIPLoss
from ColorPalette.palette import choose_colors_kmeans, read_hex , choose_colors_median_cut, choose_colors_maxmin, choose_colors_kmeans_with_rare


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Specify training stage: 1 for initial training, 2 for fine-tuning')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    return parser



# Ray marching #
@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    uncertaintys = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo_ortho.get_rays_of_a_view(              # rays_o : center #rays_d : direction
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'uncertainty', 'uncertainty_near']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [  # render views from model , input ro, rd (camera information)
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}   #return rgb_marched, depth from DVGO model
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0)) # just for efficiency enhancing
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy() # [1200,1200,1]
        depth = render_result['depth'].cpu().numpy() ## [1200,1200,1]
        bgmap = render_result['alphainv_last'].cpu().numpy() ## [1200,1200,1]
        #breakpoint()
        uncertainty = (F.elu(render_result['uncertainty_near'])+1).cpu().numpy()  ## [1200,1200,1]

        #uncertainty = F.elu(render_result['uncertainty'])+1 
        #breakpoint() # 確定一下uncertianty的範圍  (elu 後正常 1e-6~~~1)


        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        uncertaintys.append(uncertainty)

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)
    uncertaintys = np.array(uncertaintys)

    return rgbs, depths, bgmaps, uncertaintys


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)



def load_everything(args, cfg, scale_factor=1.0):
    '''Load images / poses / camera settings / data split, with scaling applied.
    Args:
        scale_factor (float): The factor to scale the scene.
    '''
    data_dict = load_data(cfg.data)

    # remove useless fields
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'depth_gt','images_gt'
    }

    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
        data_dict['images_gt'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images_gt']]
        

    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
        data_dict['images_gt'] = torch.FloatTensor(data_dict['images_gt'], device='cpu')


    # Scale poses and bounds
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['poses'][:, :3, 3] *= scale_factor  # Scale the translation component
    data_dict['near'] *= scale_factor             # Scale near bound
    data_dict['far'] *= scale_factor              # Scale far bound

    # Scale render poses if present
    if 'render_poses' in data_dict:
        data_dict['render_poses'] = torch.Tensor(data_dict['render_poses'])
        data_dict['render_poses'][:, :3, 3] *= scale_factor
    

    print(f"Scene scaled by a factor of {scale_factor}")
    return data_dict



# ------------- Compute BBox ---------------- #
# ------------- Compute BBox ---------------- #
# ------------- Compute BBox ---------------- #

def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_ortho.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_ortho.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)


    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
   
    return xyz_min, xyz_max

# --------------- return xyz_min, xyz_max


# Create a DVGO model 
def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')

    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model_class = dcvgo.DirectContractedVoxGO
        model = model_class(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:  #### 00000  Normal
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model_class = dvgo_ortho.DirectVoxGO
        model = model_class(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    # model = nn.DataParallel(model).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer


def load_existed_model(args, cfg, cfg_train, reload_ckpt_path, color_palette):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        #breakpoint()
        model_class = dvgo_ortho.DirectVoxGO
    color_num = color_palette.shape[0]
    model = utils.load_model(model_class, reload_ckpt_path,  extra_kwargs={"color_palette": color_palette,"color_num": color_num,}).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    #breakpoint()
    return model, optimizer, start





def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):

    #### ---------------------- train -------------   ###
    print(" ------------------- Start Scene Reconstruction -------------- ")

    # 初始化損失記錄
    loss_history = {
        "total_loss": [],
        "mse_loss": [],
        "alpha_loss": [],
        "depth_loss": [], 
        "clip_loss" : [],
        "clip_alpha_loss": [],
    }

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    depth_gt, HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images  , images_gt= [
        data_dict[k] for k in [
            'depth_gt', 'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'images_gt'
        ]
    ]



    #breakpoint()
    # depth_gt  (6, 1200, 1200, 1)
    # image ([18, 1200, 1200, 3])   
    # breakpoint()
    # 手動放大場景範圍
    scale_factor = 0.036 #!   #scene_scale #越小越精細

    # 將 xyz_min 和 xyz_max 轉換為 Tensor
    xyz_min = torch.tensor(xyz_min, dtype=torch.float32, device=device)
    xyz_max = torch.tensor(xyz_max, dtype=torch.float32, device=device)

    # 計算中心點和擴展範圍
    center = (xyz_min + xyz_max) / 2
    extent = (xyz_max - xyz_min) * scale_factor / 2
    xyz_min = center - extent
    xyz_max = center + extent

    print("Expanded xyz_min:", xyz_min)
    print("Expanded xyz_max:", xyz_max)
    
    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:    ###### ft_path =  reload ckpt
        reload_ckpt_path = args.ft_path
        print("reload_ckpt_path from : ", args.ft_path)
        #breakpoint()
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None


    def gather_training_rays():

        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        
        
        if data_dict['irregular_shape']:
            ori_mesh_img = [images_gt[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            ori_mesh_img = images_gt[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_ortho.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_ortho.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:  # ------------------ normal training rays  ----------------   #####  -------- random
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_ortho.get_training_rays(    ###### 這邊用到dvgo.6views裡面 把六面中心點跟射線算出來的方法
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo_ortho.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        is_transparent = (rgb_tr == 1).all(dim=-1)  # Shape: [N, H, W], 判斷每個像素是否透明    #[6, 1200, 1200]
        #is_transparent_gt = (rgb_tr_gt_ori == 1).all(dim=-1)
        #breakpoint()

        # rgb_tr [6, 1200, 1200, 3]
        # rays_o_tr [6, 1200, 1200, 3]
        # rays_d_tr [6, 1200, 1200, 3]
        # viewdirs_tr [6, 1200, 1200, 3]
        # imsz  [1, 1, 1, 1, 1, 1]
        #breakpoint()
        return is_transparent, rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler, ori_mesh_img
    
    def gather_training_rays_one(index):
        target_idx = index  # 只用第 5 張圖片

        if data_dict['irregular_shape']:
            rgb_tr_ori = images[target_idx].unsqueeze(0).to('cpu' if cfg.data.load2gpu_on_the_fly else device)
            ori_mesh_img = images_gt[target_idx].unsqueeze(0).to('cpu' if cfg.data.load2gpu_on_the_fly else device)
        else:
            rgb_tr_ori = images[i_train[target_idx]].unsqueeze(0).to('cpu' if cfg.data.load2gpu_on_the_fly else device)
            ori_mesh_img = images_gt[i_train[target_idx]].unsqueeze(0).to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_ortho.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train[target_idx]].unsqueeze(0),
            HW=np.expand_dims(HW[i_train[target_idx]], axis=0),
            Ks=np.expand_dims(Ks[i_train[target_idx]], axis=0),

            ndc=cfg.data.ndc,
            inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x,
            flip_y=cfg.data.flip_y
        )

        index_generator = dvgo_ortho.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        is_transparent = (rgb_tr == 1).all(dim=-1)  # Shape: [1, H, W]

        #breakpoint()
        # rgb_tr   [1, 1200, 1200, 3]
        # rays_o_tr [1, 1200, 1200, 3]
        # rays_d_tr [1, 1200, 1200, 3]
        # viewdirs_tr [1, 1200, 1200, 3]
        # ori_mesh_img [1, 1200, 1200, 3]
        
        
        # 將原始輸出張量複製 6 份，變成 [6, 1200, 1200, 3]
        rgb_tr = rgb_tr.repeat(6, 1, 1, 1)
        rays_o_tr = rays_o_tr.repeat(6, 1, 1, 1)
        rays_d_tr = rays_d_tr.repeat(6, 1, 1, 1)
        viewdirs_tr = viewdirs_tr.repeat(6, 1, 1, 1)
        ori_mesh_img = ori_mesh_img.repeat(6, 1, 1, 1)
        is_transparent = is_transparent.repeat(6, 1, 1)  # [1, H, W] -> [6, H, W]

        return is_transparent, rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler, ori_mesh_img

    
    
    
    def log_logit_distribution(model, step, save_dir="logit_stats", sample_voxels=100):
        os.makedirs(save_dir, exist_ok=True)

        logit_grid = model.logit.grid.detach().cpu()[0]  # [8, D, H, W]

        C, D, H, W = logit_grid.shape

        # 隨機抽 sample_voxels 個 voxel index
        flat_indices = torch.randperm(D * H * W)[:sample_voxels]
        z = flat_indices // (H * W)
        y = (flat_indices % (H * W)) // W
        x = flat_indices % W

        # [sample_voxels, C]
        sampled_logits = logit_grid[:, z.cpu(), y.cpu(), x.cpu()].T  # transpose 成 [N, 8]

        # 計算 softmax（可選用 gumbel-softmax，但這裡看 softmax 就夠）
        probs = F.softmax(sampled_logits, dim=-1)  # [N, 8]
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)  # [N]
        avg_entropy = entropy.mean().item()

        torch.save({
            "step": step,
            "logits": sampled_logits,
            "probs": probs,
            "entropy": entropy,
            "avg_entropy": avg_entropy
        }, os.path.join(save_dir, f"logits_step{step:06d}.pt"))

        print(f"📉 Saved logit distribution at step {step}, avg entropy = {avg_entropy:.4f}")
    
    def draw_logit_entropy_history(logit_dir, save_path):
        steps = []
        entropies = []

        for fname in sorted(os.listdir(logit_dir)):
            if fname.endswith(".pt") and "logits_step" in fname:
                path = os.path.join(logit_dir, fname)
                data = torch.load(path)
                steps.append(data["step"])
                entropies.append(data["avg_entropy"])

        if not steps:
            print(f"⚠️ No .pt files found in {logit_dir}")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(steps, entropies, marker='o')
        plt.xlabel("Global Step")
        plt.ylabel("Avg Entropy of Sampled Voxels")
        plt.title("Logit Sharpness Over Training")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved logit sharpness plot to: {save_path}")
        

    
    def entropy_loss(logits):
        probs = torch.softmax(logits, dim=1)
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))

    ### -------- 蒐集到射線需要的資訊  ---------- #
    
    color_num = 6  ### 調色盤顏色數量
    
    is_transparent, rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler, ori_mesh_tr = gather_training_rays()
    is_transparent_ft, rgb_tr_ft, rays_o_tr_ft, rays_d_tr_ft, viewdirs_tr_ft, _, batch_index_sampler, ori_mesh_tr_ft = gather_training_rays_one(index = 3)

    # 原始 rgb_tr: (6, 1200, 1200, 3)
    front_view_tr = rgb_tr[3]  # shape: (1200, 1200, 3)

    # 重複三次 front view，變成 (3, 1200, 1200, 3)
    #repeated_front = front_view_tr.unsqueeze(0).repeat(3, 1, 1, 1)

    # 將原本的 rgb_tr 與 repeated_front 接起來
    #rgb_tr_augmented = torch.cat([rgb_tr, repeated_front], dim=0)  # shape: (9, 1200, 1200, 3)
    
    # 接下來用 augmented tensor 做 color selection
    #color_palette = choose_colors_kmeans(rgb_tr_augmented, color_num)
    #color_palette = choose_colors_kmeans(rgb_tr, color_num)
    color_palette = choose_colors_kmeans_with_rare(rgb_tr, color_num)
    
    
        #breakpoint()
    ### Try get the color only use front view
    
    #color_palette = choose_colors_median_cut(rgb_tr, color_num)
    #color_palette = choose_colors_maxmin(rgb_tr, color_num)
    
    #breakpoint()
    #color_palette = read_hex ("ColorPalette/palettes/other/aqua_verde.hex") ### 指定顏色
    #breakpoint()
    #coarse_output_path = args.export_coarse_only
    a = cfg.basedir ##  './logs/edify/cowboy_wolf'
    b = cfg.expname ##   'dvgo_cowboy_wolf6_views'
    save_dir = os.path.join(a, b)
    os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在
    np.savez(os.path.join(save_dir, "color_palette.npz"), color_palette=color_palette)
    #breakpoint()
    
    
    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        print("----------------- create new model -------------------")
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path) ### model -> voxel grid
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print("----------------- reload exist model -------------------")
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        ### model.k0.grid.shape ###
        ### [1, 3, 40, 40, 40] ###
        
        ### p model.k0 ###
        ### DenseGrid(channels=3, world_size=[40, 40, 40]) ###
        
        # p model.density.grid.shape
        # torch.Size([1, 1, 40, 40, 40])
        
        #convert_densegrid_to_logitgrid
        
        # 如果 k0 是 DenseGrid，進行轉換
        # 使用 color_palette 初始化 logit_grid
       
            # 建立新的 LogitGrid 並替換
        ### logit_grid.shape : [1, 8, 40, 40, 40]
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path, color_palette)  ## model   ### this time model.palette is None
        for i, group in enumerate(optimizer.param_groups):
            print(f"\n[Parameter Group {i}] lr={group['lr']}")
            for p in group['params']:
                print("  param id:", id(p), "| shape:", p.shape, "| requires_grad:", p.requires_grad)

        
        
        # breakpoint()
         
        logit_grid = convert_densegrid_to_logitgrid(
                densegrid=model.k0.grid,           # [1, 3, D, H, W]
                palette=color_palette.tolist(),    # [[r,g,b], ...]
                temperature=0.1                    # 可調整溫度參數
        )
        
        
        """
        
        ### random initialize
        logit_grid = random_densegrid_to_logitgrid(
                densegrid=model.k0.grid,           # [1, 3, D, H, W]
                palette=color_palette.tolist(),    # [[r,g,b], ...]
                temperature=0.1                    # 可調整溫度參數
        )
        """
        
        
        #breakpoint()
        
        model.color_palette = torch.tensor(color_palette, dtype=torch.float32, device=device)

        logit_grid_tensor = torch.tensor(logit_grid, dtype=torch.float32, device=model.logit.grid.device)
        model.logit.grid.data.copy_(logit_grid_tensor)
        
        optimizer = torch.optim.Adam([
            {'params': model.k0.parameters()},
            {'params': model.density.parameters()},
            {'params': model.logit.parameters(), 'lr': 1e-2},
        ], lr=1e-3)
        
        """
        for i, group in enumerate(optimizer.param_groups):
            print(f"\n[Param Group {i}] lr={group['lr']}, num_params={len(group['params'])}")
            for p in group['params']:
                print("  param id:", id(p), "| shape:", p.shape)

        breakpoint()
        # [1, 3, 60, 60, 60] #[1, 1, 60, 60, 60] #[1, 8, 60, 60, 60]
        
        
        """
        
        
        #D, H, W = model.k0.grid.shape[2:]
 
        
        
        
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)

    # 重設 optimizer 為重新訓練準備
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=start)
    start = 0  # 訓練從頭開始
    
    for i, group in enumerate(optimizer.param_groups):
            print(f"\n[Param Group {i}] lr={group['lr']}, num_params={len(group['params'])}")
            for p in group['params']:
                print("  param id:", id(p), "| shape:", p.shape)
    #breakpoint()
    # init rendering setup
    
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler   #### 做另外一個gather_training_rays來做images_gt的切法？
    ### ------- find color palette ---- ###
    
    
    
    #breakpoint()
    
    
    ### 這邊算一個 整張圖的版本
    # is_transparent, rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    
    #breakpoint()
    ### breakpoint   ####check rgb_tr ##add rgb_tr_gt

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(       ##### ------------ 
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        #breakpoint()
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # ----------------------------- TRAIN DVGO ------------------------- #
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    
    
    
    ####### ----------------------- Travese All Iterations / 200000 ---------------------- #

    for global_step in trange(1+start, 1+cfg_train.N_iters):   
        

        clip_stage = True
        #clip_stage = True

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        cfg_train.pg_scale = []
        
        ### 確定一下4500後的training是否會影響geometry
        
        
        if global_step > 4500 :#4500:  ## 5000
            rgb_tr = rgb_tr_ft
            rays_o_tr = rays_o_tr_ft
            rays_d_tr = rays_d_tr_ft
            viewdirs_tr = viewdirs_tr_ft
            ori_mesh_tr = ori_mesh_tr_ft
        
        
            
        if clip_stage == True : 
     
            if cfg_train.ray_sampler == 'random':
                #print("random")
                #breakpoint()
                sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
                sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
                sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
                target = rgb_tr[sel_b, sel_r, sel_c]      ###[8192.3] 個像素真實顏色
                rays_o = rays_o_tr[sel_b, sel_r, sel_c]   ####  隨機選幾個   # [6, 1200, 1200, 3]   ->   [8192, 3]
                rays_d = rays_d_tr[sel_b, sel_r, sel_c]
                viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
                #depth_target = depth_gt[sel_b, sel_r, sel_c, 0] 
            else:
                raise NotImplementedError
            
        #else :  # clip_stage = True # epoch 100
            
        ### ------- 把它改成每100 epoch 算一次CLIP loss
        ### ------- 可能需要縮小圖片？(1200x1200都volume rend太大了) down sample之類的，或是有很多patch
            H, W = ori_mesh_tr.shape[1:3]
            patch_size = 80
            patches_per_img = 1

            all_target_patches = []
            all_rays_o_patches = []
            all_rays_d_patches = []
            all_viewdirs_patches = []

            for img_idx in range(6):  # 對每張圖片
                for _ in range(patches_per_img):
                    # 隨機選擇 patch 左上角位置
                    i = random.randint(0, H - patch_size)
                    j = random.randint(0, W - patch_size)

                    # 擷取 patch，shape: [60, 60, 3]
                    target_patch = ori_mesh_tr[img_idx, i:i+patch_size, j:j+patch_size, :]
                    rays_o_patch = rays_o_tr[img_idx, i:i+patch_size, j:j+patch_size, :]
                    rays_d_patch = rays_d_tr[img_idx, i:i+patch_size, j:j+patch_size, :]
                    viewdirs_patch = viewdirs_tr[img_idx, i:i+patch_size, j:j+patch_size, :]

                    # reshape 成 [3600, 3]
                    all_target_patches.append(target_patch.reshape(-1, 3))
                    all_rays_o_patches.append(rays_o_patch.reshape(-1, 3))
                    all_rays_d_patches.append(rays_d_patch.reshape(-1, 3))
                    all_viewdirs_patches.append(viewdirs_patch.reshape(-1, 3))

                # 合併所有 patch → [6*6*3600 = 129600, 3]
            target_ms = torch.cat(all_target_patches, dim=0)
            rays_o_ms = torch.cat(all_rays_o_patches, dim=0)
            rays_d_ms = torch.cat(all_rays_d_patches, dim=0)
            viewdirs_ms = torch.cat(all_viewdirs_patches, dim=0)
            """
            # 逐張儲存
            for i in range(target_ori_np.shape[0]):
                img_path = os.path.join(save_dir, f"image_{i}.png")  # 儲存為 PNG 格式
                imageio.imwrite(img_path, (target_ori_np[i] * 255).astype("uint8"))  # 轉換為 0-255
                print(f"Saved {img_path}")
            """

            #breakpoint()
        


        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)


        if clip_stage == True:
            # volume rendering
            render_result = model(   ####DVGO.forward     ####Dict    #####整個DVGO是一個dict
                rays_o, rays_d, viewdirs,
                global_step=global_step, is_train=True,
                **render_kwargs)
        #else: 
            render_result_clip = model(   ####DVGO.forward     ####Dict    #####整個DVGO是一個dict
                rays_o_ms, rays_d_ms, viewdirs_ms,
                global_step=global_step, is_train=True,
                **render_kwargs)
            
            #breakpoint()

        optimizer.zero_grad(set_to_none=True)
        
        """
        if global_step==5:
            for i, group in enumerate(optimizer.param_groups):
                print(f"Group {i} - lr: {group['lr']}")
                for p in group['params']:
                    print(f"  Param shape: {p.shape}, requires_grad: {p.requires_grad}")

            #breakpoint()
        
        if global_step>50:
            for i, group in enumerate(optimizer.param_groups):
                print(f"Group {i} - lr: {group['lr']}")
                for p in group['params']:
                    print(f"  Param shape: {p.shape}, requires_grad: {p.requires_grad}")

            #breakpoint()
        """
        
        # --------------------Uncertainty & MSE  loss -------------------- #
        # gradient descent step
        if clip_stage == True : 
            # 找出 target 中 [1, 1, 1] 的像素索引
            transparent_mask = (target == 1).all(dim=-1)  # 形狀為 [8192]，每個值為 True 或 False
            mse_loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)     #[8192, 3]

            #### ------ loss
        
            ####### 8192個經過volume rendering 的 pixel 各個rgb值  ### 跟 rgb_tr 經過random切割後[8192]

            # ------------------- Trans Pixel Alpha Loss ------------- #
            # 提取透明像素的 alphainv_last
            transparent_alpha = render_result['alphainv_last'][transparent_mask]  # Shape: [M]

            # 懲罰透明像素中未完全穿透的 alpha
            transparent_loss = ((1 - transparent_alpha) ** 2).mean()  # 1代表完全穿透
            trans_loss_weight = 10.0 #5.0 #1.5
            #1.5  ##1.5   #### 1    ###

            # ------------------ Depth Loss ----------------- #
            # 根據隨機選取的射線索引，提取對應的深度 GT
            depth_gt = torch.tensor(depth_gt, device=device)
    
            depth_target = depth_gt[sel_b.to(depth_gt.device), sel_r.to(depth_gt.device), sel_c.to(depth_gt.device), 0]
            

            render_depth  = render_result['depth']    # 形狀為[8192]， min : 0.0, max : 81.49 
            depth_target = depth_target * (render_depth.mean() / depth_target.mean()) # Normalize
            
            render_depth_nor = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
            depth_target_nor = (depth_target - depth_target.min()) / (depth_target.max() - depth_target.min())

            depth_loss = torch.nn.functional.smooth_l1_loss(render_depth_nor, depth_target_nor)
        
            if(cfg_model.num_voxels <= 110592):  ### >= cell size 25con
                depth_weight = 10 #20 #5 #5 #1 #1 #0.5 #2 #1
            else:
                depth_weight = 20 #70 #40 #10 #10 #5 #10  ###   

            if (global_step>4500):
                depth_weight = 30 ## 80

            depth_loss *= depth_weight
            transparent_loss = transparent_loss * trans_loss_weight
            
            mse_loss *= 5
            loss_ori = mse_loss  +  transparent_loss + depth_loss  ## mse * 3 
            
            
            #breakpoint()
            
            
            #loss = mse_loss
            # 記錄每個損失
            #loss_history["total_loss"].append(loss_ori.item())
            loss_history["mse_loss"].append(mse_loss.item())
            loss_history["alpha_loss"].append(transparent_loss.item())
            loss_history["depth_loss"].append(depth_loss.item())

            # print( " ------ scene_rep_reconstru ------ ")
            # 使用 .detach() 來移除梯度追蹤
            # np.savez_compressed("scripts/render_depth_qqqq.npz", render_depth = depth_target.detach().cpu().numpy())
            # np.savez_compressed("scripts/depth_target_qqqq.npz", depth_target = render_depth.detach().cpu().numpy())
            #breakpoint()
            #mse_loss = F.mse_loss(render_result['rgb_marched'], target)
            
            psnr = utils.mse2psnr(mse_loss.detach())
            
            if cfg_train.weight_entropy_last > 0:
                pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
                entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                loss_ori += cfg_train.weight_entropy_last * entropy_last_loss
            if cfg_train.weight_nearclip > 0:
                near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
                near_mask = (render_result['t'] < near_thres)
                density = render_result['raw_density'][near_mask]
                if len(density):
                    nearclip_loss = (density - density.detach()).sum()
                    loss_ori += cfg_train.weight_nearclip * nearclip_loss
            if cfg_train.weight_distortion > 0:
                n_max = render_result['n_max']
                s = render_result['s']
                w = render_result['weights']
                ray_id = render_result['ray_id']
                loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
                loss_ori += cfg_train.weight_distortion * loss_distortion
            if cfg_train.weight_rgbper > 0:
                rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
                rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                loss_ori += cfg_train.weight_rgbper * rgbper_loss


        #else: 
            
        
            transparent_mask_clip = (target_ms == 1).all(dim=-1) 
            # 提取透明像素的 alphainv_last
            transparent_alpha_clip = render_result_clip['alphainv_last'][transparent_mask_clip]  # Shape: [M]

            # 懲罰透明像素中未完全穿透的 alpha
            transparent_loss_clip = ((1 - transparent_alpha_clip) ** 2).mean()  # 1代表完全穿透
            trans_loss_weight = 4 #2.0 #3.0 #5.0 #1.5
            transparent_loss_clip = transparent_loss_clip * trans_loss_weight
            
            #breakpoint()
            if (global_step<6001):
                clip_loss_fn = CLIPLoss(device="cuda", save_dir="test_img")
                
                #clip_loss = loss_ori
                
                patch_size = 80
                num_patches = render_result_clip['rgb_marched'].shape[0] // (patch_size * patch_size)

                render_rgb = render_result_clip['rgb_marched'].reshape(num_patches, patch_size, patch_size, 3)
                target_rgb = target_ms.reshape(num_patches, patch_size, patch_size, 3)
                
                clip_loss = clip_loss_fn(render_rgb, target_rgb)
            
                #loss =   loss_ori + transparent_loss_clip + clip_loss
                loss =   2 * loss_ori   + 1 *(transparent_loss_clip + clip_loss) # 0.1
            else : 
                loss = 10 * transparent_loss
                clip_loss = torch.tensor(0.0, device="cuda")
                transparent_loss_clip = torch.tensor(0.0, device="cuda")
            #loss =  2 * loss_ori
            
            
            loss_history["clip_loss"].append(clip_loss.item())
            loss_history["clip_alpha_loss"].append(transparent_loss_clip.item())
            loss_history["total_loss"].append(loss.item())


        loss.backward()
        #breakpoint()
        ## p model.logit.grid.shape    # torch.Size([1, 8, 60, 60, 60])
        ## p model.k0.grid.grad.shape  # torch.Size([1, 3, 60, 60, 60])
        
        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(    
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_uncertainty>0:
                model.uncertainty_total_variation_add_grad(
                    cfg_train.weight_tv_uncertainty/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        
        #psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
        
        
  
        #breakpoint()

        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.4f}' f'MSE Loss:{mse_loss.item()*3:.4f}' f'Depth Loss: {depth_loss.item():.4f}'  f'CLIP Loss: {clip_loss.item():.4f}' 
                       f'Eps: {eps_time_str}')
        
            
        if global_step % 500 == 0:
            log_logit_distribution(
                model=model,
                step=global_step,
                save_dir=os.path.join(cfg.basedir, cfg.expname, "logit_stats")
            )



    if global_step != -1 :
  
        model_state = model.state_dict()

        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)

        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
    
    """---- saving density ----"""
    



    #density = render_result['raw_density'][near_mask]
    

    # 繪製損失曲線
    def smooth_curve(data, window_size=100):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_curve(loss_history["total_loss"], window_size=100), label=f"Total Loss (Smoothed)", color="blue", alpha=0.7)
    plt.plot(smooth_curve(loss_history["mse_loss"], window_size=100), label=f"MSE Loss w = 1", color="red", alpha=0.7)
    plt.plot(smooth_curve(loss_history["alpha_loss"], window_size=100), label=f"Alpha Loss w = {trans_loss_weight}", color="green", alpha=0.7)
    plt.plot(smooth_curve(loss_history["depth_loss"], window_size=100), label=f"Depth Loss w = {depth_weight}", color="orange", alpha=0.7)
    plt.plot(smooth_curve(loss_history["clip_loss"], window_size=100), label=f"Clip Loss w = {depth_weight}", color="yellow", alpha=0.7)
    plt.plot(smooth_curve(loss_history["clip_alpha_loss"], window_size=100), label=f"Clip Alpha Loss w = {depth_weight}", color="purple", alpha=0.7)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss curve (depth loss weight = {depth_weight} )")  # 新增 title，顯示 depth loss weight = 0
    plt.legend()
    plt.grid()
    plt.tight_layout()


    # 保存圖表
    loss_plot_path = os.path.join(cfg.basedir, cfg.expname, f"loss_curve_{stage}_clip.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    draw_logit_entropy_history(
    logit_dir=os.path.join(cfg.basedir, cfg.expname, "logit_stats"),
    save_path=os.path.join(cfg.basedir, cfg.expname, "logit_his.png")
    )

    
    #np.savez("depth_data.npz", render_depth=render_depth_np, depth_target=depth_target_np)
    #print("save at the end")
    #breakpoint()


def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    
    #xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    xyz_min_coarse = torch.tensor([-10, -10, -10])
    xyz_max_coarse = torch.tensor([10,   10,  10])

    # save coarse training ckpt
    if cfg.coarse_train.N_iters > 0:
        #breakpoint()
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    print('train: skip fine detail reconstruction')
    
    """
    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo_ortho.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)
    
    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')
    """

 
if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split /
    data_dict = load_everything(args=args, cfg=cfg)    ####  data_dict HERE
    print(" ----- main.py ----- ")
    #print(" After load everythings ")

    # export scene bbox and camera poses in 3d for debugging and visualization
    # ------ get camera rays line visualization ----------# 
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        #xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        xyz_min = tensor([-10, -10, -10])
        xyz_max = tensor([10, 10, 10])
        xyz_max = torch.tensor(xyz_max)
        xyz_min = torch.tensor(xyz_min)
        print("xyz_min = ", xyz_min)
        print("xyz_max = ", xyz_max)

        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']

        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo_ortho.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy() # only save a rays of rays_o(pixels) 
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy() 
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))

        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=cam_lst)
        print('done')

        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')   #logs/nerf_synthetic/dvgo_cat_ortho/coarse_last.tar
            model = utils.load_model(dvgo_ortho.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            #rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
            rgb = torch.sigmoid(model.logit.get_logit_grid()).squeeze().permute(1,2,3,0).cpu().numpy()   #### save logit
            uncertainty = model.activate_density(model.uncertainty.get_dense_grid()).squeeze().cpu().numpy()

        # ----- save ------ #
        coarse_output_path = args.export_coarse_only
        os.makedirs(os.path.dirname(coarse_output_path), exist_ok=True)
        np.savez_compressed(coarse_output_path, alpha=alpha, rgb=rgb, uncertainty=uncertainty)   ##### alpha,rgb
        print('done')
        # ---------------- #
        print("alpha shape = ", alpha.shape)  #alpha shape =  (105, 104, 92)
        print("rgb shape = ", rgb.shape) # rgb shape =  (105, 104, 92, 3)
        sys.exit()

    # train

    ###### ,  這裡train -> train 包 scene_recons  #傳入data_dict
    if not args.render_only:
        train(args, cfg, data_dict)   # ------------- train 
  
    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        """
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            #ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar') #fine training
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
        """
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]

        
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo_ortho.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        
        
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }
    
    
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, uncertaintys = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)

        """
        # Render the test set
        rgbs, depths, bgmaps, uncertaintys = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir, dump_images=args.dump_images,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs
        )

        # Save RGB video
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)

        # Save depth video
        normalized_depths = 1 - depths / np.max(depths)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(normalized_depths), fps=30, quality=8)

        # Save uncertainty video
        normalized_uncertaintys = 1 - uncertaintys / np.max(uncertaintys)
        imageio.mimwrite(os.path.join(testsavedir, 'video.uncertainty.mp4'), utils.to8b(normalized_uncertaintys), fps=30, quality=8)

        uncertainty_save_path = os.path.join(testsavedir, 'uncertainty_map.npz')
        np.savez_compressed(uncertainty_save_path, uncertainty=uncertaintys)
        print(f"Uncertainty maps saved to {uncertainty_save_path}")
        """
        # Save depths as .npz file
        """
        depth_save_path = os.path.join(testsavedir, 'test_depths.npz')
        np.savez_compressed(depth_save_path, depths=depths)
        print(f"Depth maps saved to {depth_save_path}")
        """

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, uncertaintys = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,

                **render_viewpoints_kwargs)
        
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

         # ✅ 存 Uncertainty 影片
        uncertaintys_vis = uncertaintys * (1 - bgmaps) + bgmaps
        dmin, dmax = np.percentile(uncertaintys_vis[bgmaps < 0.1], q=[5, 95])
        uncertainty_vis = plt.get_cmap('rainbow')(1 - np.clip((uncertaintys_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.uncertainty.mp4'), utils.to8b(uncertainty_vis), fps=30, quality=8)

    print('Done')
