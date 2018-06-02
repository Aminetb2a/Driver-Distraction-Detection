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
K.set_image_dim_ordering('th')




def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture_vgg166.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights_vggg.h5'), overwrite=True)
    
def VGG_16_Model():
    # build the VGG16 model
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


# def VGG_19_Model():
#     img_width, img_height = 180,180
#     # build the VGG19 model
#     model = Sequential()
#     model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
#     model.add(Conv2D(64, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(64,(3, 3), activation="relu"))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#     print('Block1')

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(128, (3, 3), activation="relu"))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#     print('Block2')

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#     print('Block3')

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#     print('model4')

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation="relu"))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
#     print('Block5')

#     '''
#     # load the weights of the VGG16 networks (trained on ImageNet, won the ILSVRC competition in 2014)
#     # note: when there is a complete match between your model definition
#     # and your weight savefile, you can simply call model.load_weights(filename)
#     '''
#     # load the weights for each layer
#     weights_path = 'vgg19_weights_th_dim_ordering_th_kernels_notop.h5'

#     model.load_weights(weights_path)

#     print('VGG16 model weights have been successfully loaded.')

#     # build a MLP classifier model to put on top of the VGG16 model
#     top_model = Sequential()
#     # flateen the output of VGG16 model to 2D Numpy matrix (n*D)
#     top_model.add(Flatten(input_shape=model.output_shape[1:]))
#     # hidden layer of 256 neurons
#     top_model.add(Dense(256, activation='relu'))
#     # add dropout for the dense layer
#     top_model.add(Dropout(0.5))
#     # the output layer: we have 10 claases
#     top_model.add(Dense(10, activation='softmax'))

#     # connect the two models onto the VGG16 net
#     model.add(top_model)


#     # # set the first 25 layers (up to the last conv block) of VGFG16 net to non-trainable (weights will not be updated)
#     # for layer in model.layers[:25]:
#     #     layer.trainable=False

#     # compile the model 
#     model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
#     print ("the model has been successfully created")
#     return model

def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model



use_cache = 1

''' path to the model weights file in HDF5 binary data format
The vgg16 weights can be downloaded from the link below:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
'''

# dimensions of the images
img_width, img_height = 180, 180

# the path to the training data
train_data_dir = 'data/train'

# the path to the validation data
validation_data_dir = 'data/validation'

# the number of training samples. We have 20924 training images, but actually we can set the 
# number of training samples can be augmented to much more, for example 2*20924
nb_train_samples = 22450

# We actually have 1500 validation samples, which can be augmented to much more
nb_validation_samples = 4066

# number of epoches for training
nb_epoch = 20

model = VGG19(include_top=False, weights='imagenet')
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
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
with open("model_vgg19.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_vgg19.h5")
print("Saved model to disk")










            

# augmentation configuration for training data
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
# fit the model
# hist = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, 
#                     validation_data=validation_generator, nb_val_samples=nb_validation_samples)

# hist = model.fit_generator(gen.generate(True), samples_per_epoch=gen.train_batches, nb_epoch=nb_epoch, 
#                      validation_data=gen.generate(False), nb_val_samples=gen.val_batches)

# hist = model.fit_generator(gen.generate(True), gen.train_batches,
#                               epochs=nb_epoch, verbose=1,
                              
#                               validation_data=gen.generate(False),
#                               validation_steps=gen.val_batches,
#                               )
# fit_generator(<generator..., steps_per_epoch=1703, validation_steps=426, validation_data=<generator..., epochs=10)