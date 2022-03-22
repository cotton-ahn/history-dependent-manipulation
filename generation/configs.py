import numpy as np


rot_thres = 7.5
random_rot = 3
block_size = 0.5
eps = 0.2
pos_bound = 3.2
empty_prob = 0.25
max_dependency = 4

positions = dict()
positions['left'] = np.asarray([-2.5, 0.0, 0.5])
positions['right'] = np.asarray([2.5, 0.0, 0.5])
positions['front'] = np.asarray([0.0, -2.5, 0.5])
positions['rear'] = np.asarray([0.0, 2.5, 0.5])
positions['front_left'] = np.asarray([-2.5, -2.5, 0.5])
positions['front_right'] = np.asarray([2.5, -2.5, 0.5])
positions['rear_left'] = np.asarray([-2.5, 2.5, 0.5])
positions['rear_right'] = np.asarray([2.5, 2.5, 0.5])
positions['center'] = np.asarray([0.0, 0.0, 0.5])

strides = dict()
stride_len = 1.0 + 0.06
strides['left'] = np.asarray([-stride_len, 0.0, 0.0])
strides['right'] = np.asarray([stride_len, 0.0, 0.0])
strides['front'] = np.asarray([0.0, -stride_len, 0.0])
strides['rear'] = np.asarray([0.0, stride_len, 0.0])
strides['front_left'] = np.asarray([-1.001, -1.001, 0.0])
strides['front_right'] = np.asarray([1.001, -1.001, 0.0])
strides['rear_left'] = np.asarray([-1.001, 1.001, 0.0])
strides['rear_right'] = np.asarray([1.001, 1.001, 0.0])
strides['upper'] =  np.asarray([0.0, 0.0, 1.0])
