_base_ = '../default_6views.py'

expname = 'dvgo_corgi6_views'
basedir = './logs/Check/corgi'

data = dict(
    datadir='./data/Check/corgi/6views',
    dataset_type='blender',
    white_bkgd=True,
)

