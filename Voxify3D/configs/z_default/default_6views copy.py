from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs
num_voxels_ini = 64000
''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    datatype='bounded',
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    rand_bkgd=False,              # use random background during training
    half_res=False,               # [TODO]
    bd_factor=.75,
    movie_render_kwargs=dict(),

    # Below are forward-facing llff specific settings.
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    factor=4,                     # [TODO]
    width=None,                   # enforce image width
    height=None,                  # enforce image height
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    # Below are unbounded inward-facing specific settings.
    unbounded_inward=False,   
    unbounded_inner_r=1.0,
)

''' Template of training options
'''
coarse_train = dict(
    N_iters=12000,  #6000
    N_rand=8192,
    lrate_density= 0.04, #1e-1, #0.04,#0.04,
    lrate_k0=0.005, ### 0.005不能再高了 
    lrate_uncertainty = 1e-1, #5e-1, #1e-1
    lrate_logit = 1e-1, #1e-1, #5e-2, #1e-1, #5e-1, #1e-1 ##小丑30試一下5e-2, 
    lrate_rgbnet=0.005,  #基本上沒差
    lrate_decay=20, #20,
    pervoxel_lr=False, # pervoxel_lr
    pervoxel_lr_downrate=1,
    ray_sampler='random',
    weight_main=1.0,
    weight_entropy_last=0.5,   #>0.03
    weight_nearclip=0,
    weight_distortion=0,
    weight_rgbper=5.0,
    tv_every=1,
    tv_after=0,
    tv_before=0,
    tv_dense_before=0,
    weight_tv_density= 10,
    weight_tv_k0= 1,  ##0.01 #沒差
    weight_tv_uncertainty=0.0, 
    pg_scale=[],
    decay_after_scale=1.0,
    skip_zero_grad_fields=[],
    maskout_lt_nviews=0,
    )

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=5000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    # pg_scale=[1000, 2000, 3000, 4000],
    pg_scale=[],
    skip_zero_grad_fcields=['density', 'k0'],
    uncertainty=False,
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=num_voxels_ini,          #3072000 -> 3 voxel...   #10240000 -> 9....       # expected number of voxel #1024000  # 1024只有幾塊。。
    num_voxels_base=1000000,      # to rescale delta distance #1024000 #
    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
    k0_type='DenseGrid',          # DenseGrid, TensoRFGrid
    density_config=dict(),
    k0_config=dict(),
    mpi_depth=256,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest= True,    #False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=5,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=256,             # width of the colors MLP
    alpha_init=1e-2 ,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-8,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1.0,          # rescale the BBox enclosing the scene
    stepsize=0.25,                 # sampling stepsize in volume rendering
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=12, 
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.00,
))

del deepcopy