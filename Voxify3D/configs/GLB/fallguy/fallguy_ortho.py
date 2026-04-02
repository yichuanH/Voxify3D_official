_base_ = '../default_ortho.py'

expname = 'dvgo_fallguy_ortho'
basedir = './logs/GLB/fallguy'

data = dict(
    datadir='./data/GLB/fallguy/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

