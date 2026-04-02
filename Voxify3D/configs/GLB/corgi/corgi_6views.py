_base_ = '../default_6views.py'

expname = 'dvgo_corgi6_views'
basedir = './logs/GLB/corgi'

data = dict(
    datadir='./data/GLB/corgi/6views',
    dataset_type='blender',
    white_bkgd=True,
)

