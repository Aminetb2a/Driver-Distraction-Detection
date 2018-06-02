'''
This program is used to recognize the driver's status (one of the 10 statuses) based on the image using pre-trained VGG16 
deep convolutional neural network (CNN).

This program is modified from the blog post: 
"Building powerful image classification models using very little data" from blog.keras.io.

This program do fine tunning for a modified VGG16 net, which consists of two parts: 
the lower model: layer 0-layer24 of the original VGG16 net  (frozen the first 4 blocks, train the weights of the 5-th block 
with our dataset)
the upper model: newly added two layer dense net (train the weights using our dataset)
'''
import sys
import os,random
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,Conv2D
import glob
import cv2
import math
import keras
import pickle
import datetime
import pandas as pd
from shutil import *
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras import backend as K
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from distutils.version import StrictVersion
from keras import __version__ as keras_version
import time
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, RemoteMonitor, TensorBoard, ReduceLROnPlateau,LambdaCallback
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
from keras import optimizers
from keras.optimizers import SGD
K.set_image_dim_ordering('th')


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture_vgg166.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights_vggg.h5'), overwrite=True)
    
def VGG_16_Model(img_width, img_height):
    # build the VGG16 model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block11')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block2')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block3')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block4')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block5')

    weights_path = 'vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
    model.load_weights(weights_path)
    print('VGG16 model weights have been successfully loaded.')

    # build a MLP classifier model to put on top of the VGG16 model
    top_model = Sequential()
    # flateen the output of VGG16 model to 2D Numpy matrix (n*D)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # hidden layer of 256 neurons
    top_model.add(Dense(256, activation='relu'))
    # add dropout for the dense layer
    top_model.add(Dropout(0.5))
    # the output layer: we have 10 claases
    top_model.add(Dense(10, activation='softmax'))

    # connect the two models onto the VGG16 net
    model.add(top_model)
    print('Top Block')

    # set the first 25 layers (up to the last conv block) of VGFG16 net to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable=False

    # compile the model 
    model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print('Vgg_16 has been successfully created')
    return model
    
    
def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_test = []
    y_test = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_test.append(img)
            y_test.append(j)
    return X_test, y_test



def VGG_19_Model():
    img_width, img_height = 180,180
    # build the VGG19 model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block1')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block2')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block3')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('model4')

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print('Block5')

    '''
    # load the weights of the VGG16 networks (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    '''
    # load the weights for each layer
    weights_path = 'vgg19_weights_th_dim_ordering_th_kernels_notop.h5'

    model.load_weights(weights_path)

    print('VGG16 model weights have been successfully loaded.')

    # build a MLP classifier model to put on top of the VGG16 model
    top_model = Sequential()
    # flateen the output of VGG16 model to 2D Numpy matrix (n*D)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # hidden layer of 256 neurons
    top_model.add(Dense(256, activation='relu'))
    # add dropout for the dense layer
    top_model.add(Dropout(0.5))
    # the output layer: we have 10 claases
    top_model.add(Dense(10, activation='softmax'))

    # connect the two models onto the VGG16 net
    model.add(top_model)


    # set the first 25 layers (up to the last conv block) of VGFG16 net to non-trainable (weights will not be updated)
    for layer in model.layers[:28]:
        layer.trainable=False

    # compile the model 
    # model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    print ("the model has been successfully created")
    return model




use_cache = 1

''' path to the model weights file in HDF5 binary data format
The vgg16 weights can be downloaded from the link below:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
'''

# dimensions of the images
img_width, img_height = 224, 224

# the path to the training data
train_data_dir = 'train'

# the path to the validation data
validation_data_dir = 'data/validation'

# the number of training samples. We have 20924 training images, but actually we can set the 
# number of training samples can be augmented to much more, for example 2*20924
nb_train_samples = 17680

# We actually have 1500 validation samples, which can be augmented to much more
nb_validation_samples = 2200

# number of epoches for training
nb_epoch = 30

x_data,y_labels=load_train()

X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.4, random_state=0)

print('test')
model = VGG_16_Model(img_width, img_height)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 16
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# steps_per_epoch should be (number of training images total / batch_size) 
# validation_steps should be (number of validation images total / batch_size) 
hist = model.fit_generator(generator=train_generator, steps_per_epoch=nb_train_samples/batch_size, epochs=nb_epoch, 
                     validation_data=val_generator, validation_steps=nb_validation_samples/batch_size)




#Plotting the Graphs
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc'] 
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

#train loss vs. val loss
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of epochs')
plt.ylabel('loss')
plt.title('train loss vs. val loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])


#rain acc vs. val acc
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of epochs')
plt.ylabel('accuracy')
plt.title('train acc vs. val acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()

#end of Plotting


#saving the model
model_json = model.to_json()
with open("model_vgg16_2.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_vgg16_2.h5")
print("Saved model to disk")            




# # augmentation configuration for training data
# train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# print('here2')
# # augmentation configuration for validation data (actually we did no augmentation to teh validation images)
# test_datagen = ImageDataGenerator(rescale=1.0/255)
# print('here3')

# # training data generator from folder
# train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), 
#                                                   batch_size=32, class_mode='categorical')
# print('here4')
# # validation data generator from folder
# validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), 
#                                                        batch_size=32, class_mode='categorical')

# X_train = []
# X_train_id = []
# y_train = []
# input_shape = (180, 180, 3)

# gt={}
# print('Read train images')
# for j in range(10):
#     print('Load folder c{}'.format(j))
#     path = os.path.join('input','train', 'c' + str(j), '*.jpg')
#     files = glob.glob(path)
#     gt.update ({fl:j for fl in files})




# keys = sorted(gt.keys()) 

# num_train = int(round(0.8 * len(keys)))
# # prisnt(num_train)
# train_keys = keys[:num_train]

# # print('dsds',len(keys))
# val_keys = keys[num_train:]
# # print('hererehrehre',val_keys)
# num_val = len(val_keys)

# path_prefix = ''
# gen = Generator(gt, 32, path_prefix,
#                 train_keys, val_keys,
#                 (input_shape[0], input_shape[1]))


# print('here5')
# # fit the model
# hist = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, 
#                     validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# hist = model.fit_generator(gen.generate(True), samples_per_epoch=gen.train_batches, nb_epoch=nb_epoch, 
#                      validation_data=gen.generate(False), nb_val_samples=gen.val_batches)

# hist = model.fit_generator(gen.generate(True), gen.train_batches,
#                               epochs=nb_epoch, verbose=1,
                              
#                               validation_data=gen.generate(False),
#                               validation_steps=gen.val_batches,
#                               )
# # fit_generator(<generator..., steps_per_epoch=1703, validation_steps=426, validation_data=<generator..., epochs=10)ZZZZZZZZ