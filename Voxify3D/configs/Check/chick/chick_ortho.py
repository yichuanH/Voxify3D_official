_base_ = '../default_ortho.py'

expname = 'dvgo_chick_ortho'
basedir = './logs/Check/chick'

data = dict(
    datadir='./data/Check/chick/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

