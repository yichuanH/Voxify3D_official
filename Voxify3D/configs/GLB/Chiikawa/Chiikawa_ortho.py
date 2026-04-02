_base_ = '../default_ortho.py'

expname = 'dvgo_Chiikawa_ortho'
basedir = './logs/GLB/Chiikawa'

data = dict(
    datadir='./data/GLB/Chiikawa/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

