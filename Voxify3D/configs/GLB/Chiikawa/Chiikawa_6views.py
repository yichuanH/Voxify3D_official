_base_ = '../default_6views.py'

expname = 'dvgo_Chiikawa6_views'
basedir = './logs/GLB/Chiikawa'

data = dict(
    datadir='./data/GLB/Chiikawa/6views',
    dataset_type='blender',
    white_bkgd=True,
)

