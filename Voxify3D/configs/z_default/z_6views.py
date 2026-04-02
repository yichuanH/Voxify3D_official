_base_ = '../default_6views.py'

expname = 'dvgo_car6_views'
basedir = './logs/our_data/car'

data = dict(
    datadir='./data/our_data/car/6views',
    dataset_type='blender',
    white_bkgd=True,
)

