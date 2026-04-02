_base_ = '../default_ortho.py'

expname = 'dvgo_car_ortho'
basedir = './logs/our_data/car'

data = dict(
    datadir='./data/our_data/car/ortho',
    dataset_type='blender',
    white_bkgd=True,
)

