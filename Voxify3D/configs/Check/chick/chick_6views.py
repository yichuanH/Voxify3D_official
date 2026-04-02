_base_ = '../default_6views.py'

expname = 'dvgo_chick6_views'
basedir = './logs/Check/chick'

data = dict(
    datadir='./data/Check/chick/6views',
    dataset_type='blender',
    white_bkgd=True,
)

