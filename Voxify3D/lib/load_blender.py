import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import numpy as np

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    all_imgs_gt = []

    ### 把所有train, val, test image 讀進來
    # ----- imgs load start ------- #
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # ----- imgs load end ------- #
    # ----- Ground Truth (GT) imgs load start ------- #
    # 找到與 basedir 平行的資料夾名稱
    parent_dir = os.path.dirname(basedir)  # 取得 `6views` 的上層資料夾
    current_folder_name = os.path.basename(basedir)  # `6views`
    gt_folder_name = current_folder_name + "_n"  # 變成 `6views_n`
    basedir_gt = os.path.join(parent_dir, gt_folder_name, "train")  # 自動推導 `6views_n/train`
    # 檢查 GT 資料夾是否存在
    if not os.path.exists(basedir_gt):
        raise FileNotFoundError(f"Ground truth folder not found: {basedir_gt}")
    gt_imgs = []
    gt_files = sorted(os.listdir(basedir_gt))  # 確保順序一致
    for fname in gt_files:
        if fname.endswith('.png'):
            img = imageio.imread(os.path.join(basedir_gt, fname))
            gt_imgs.append(img)

    gt_imgs = (np.array(gt_imgs) / 255.).astype(np.float32)  # 標準化並轉成 float32
    all_imgs_gt.append(gt_imgs)

    imgs_gt = np.concatenate(all_imgs_gt, 0)  # 合併所有 GT 影像






    #imgs.shape = 18, 1200, 1200, 4
    # print("load_blender")
    # breakpoint()

    depth_npz_path = os.path.join(basedir, 'test_depths.npz')

    #breakpoint()
    if os.path.exists(depth_npz_path):
        depth_data = np.load(depth_npz_path)
        key = list(depth_data.keys())[0]
        depth_gt = depth_data[key]
    else:
        depth_gt = None
    

    uncertainty_npz_path = os.path.join(basedir, 'uncertainty_map.npz')
    #breakpoint()
    if os.path.exists(uncertainty_npz_path):
        uncertainty_data = np.load(uncertainty_npz_path)
        key = list(uncertainty_data.keys())[0]
        uncertainty_mask = uncertainty_data[key]

        """
        # uncertainty weight *2    
        uncertainty_mask = uncertainty_mask.astype(np.float32)
        uncertainty_mask = np.clip(uncertainty_mask * 2, 0, 1)
        """

    else:
        uncertainty_mask = None

    # load_blender

    #print("load_blender.py")
    # import ipdb; ipdb.set_trace() 
    # imgs.shape = [18, 1200, 1200, 4]


    return uncertainty_mask, depth_gt, imgs, poses, render_poses, [H, W, focal], i_split, imgs_gt


