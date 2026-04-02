_base_ = '../default_6views.py'

expname = 'dvgo_fallguy6_views'
basedir = './logs/GLB/fallguy'

data = dict(
    datadir='./data/GLB/fallguy/6views',
    dataset_type='blender',
    white_bkgd=True,
)

