import numpy as np
import os
import glob
import cv2
import pickle
import datetime
import pandas as pd
import time
from shutil import copy2
import warnings
warnings.filterwarnings("ignore")
from numpy.random import permutation
np.random.seed(2016)
from keras.models import Sequential

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,Conv2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
import h5py
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard


#load data
def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    return resized

def get_driver_data():
    dr = dict()
    clss = dict()
    path = os.path.join('input', 'driver_imgs_list2.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
        if arr[0] not in clss.keys():
            clss[arr[0]] = [(arr[1], arr[2])]
        else:
            clss[arr[0]].append((arr[1], arr[2]))
    f.close()
    return dr, clss

def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    driver_id = []
    start_time = time.time()
    driver_data, dr_class = get_driver_data()
    print(driver_data)
    print(dr_class)

    print('Read train images')
    for j in range(9):
        print('Load folder c{}'.format(j))
        path = os.path.join('input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, X_train_id, driver_id, unique_drivers
#loading data from cache
def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')
        
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data





def RunFile():
	use_cache=1
	cache_path = os.path.join('cache', 'train_r_' + str(224) + '_c_' + str(224) + '_t_' + str(5) + '.dat')
	if not os.path.isfile(cache_path) or use_cache == 0:
	    train_data, train_target, train_id, driver_id, unique_drivers = load_train()
	    cache_data((train_data, train_target, train_id, driver_id, unique_drivers), cache_path)
	else:
	    print('Restore train from cache!')
	    (train_data, train_target, train_id, driver_id, unique_drivers) = restore_data(cache_path)

	print('Convert to numpy...')
	train_data = np.array(train_data, dtype=np.uint8)
	train_target = np.array(train_target, dtype=np.uint8)
	print('Reshape...')
	train_data = train_data.transpose((0, 3, 1, 2))

	print('Convert to float...')
	train_data = train_data.astype('float16')
	mean_pixel = [103.939, 116.779, 123.68]
	print('Substract 0...')
	train_data[:, 0, :, :] -= mean_pixel[0]
	print('Substract 1...')
	train_data[:, 1, :, :] -= mean_pixel[1]
	print('Substract 2...')
	train_data[:, 2, :, :] -= mean_pixel[2]

	train_target = np_utils.to_categorical(train_target, 9)

	# Shuffle experiment START !!!
	perm = permutation(len(train_target))
	train_data = train_data[perm]
	train_target = train_target[perm]
	# Shuffle experiment END !!!
	train_data = train_data.transpose((0, 2, 3, 1))
	print('Train shape:', train_data.shape)
	print(train_data.shape[0], 'train samples')


	return train_data,train_target