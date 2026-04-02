_base_ = '../default_ortho.py'

expname = 'dvgo_corgi_ortho'
basedir = './logs/Check/corgi'

data = dict(
    datadir='./data/Check/corgi/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

