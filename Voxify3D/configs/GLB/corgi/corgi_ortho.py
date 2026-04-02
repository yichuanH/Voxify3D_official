_base_ = '../default_ortho.py'

expname = 'dvgo_corgi_ortho'
basedir = './logs/GLB/corgi'

data = dict(
    datadir='./data/GLB/corgi/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

