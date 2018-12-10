"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import data_prepare as dp
from keras.utils import plot_model
from keras.models import Model
import time

import numpy as np
import resnet

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')


batch_size = 32
nb_classes = 3
nb_epoch = 1
data_augmentation = False

# input image dimensions
img_rows, img_cols = 224, 224
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = dp.load_Data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

# # build and load weights for resnet 50
# model_50 = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
# model_50.load_weights("model/weights/50weights.h5")
#
# #build and load weights for resnet 152
# model_152 = resnet.ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)
# model_152.load_weights("model/weights/152weights.h5")

# set intermediate input shape and build second half model for reset 152
intermediate_shape = (56, 56, 256)
intermediate_shape_1 = (28, 28, 512)

output_layer = 33
input_layer = 34


model_152 = resnet.ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)
model_152.summary()

# model_152_second_half_2 = resnet.ResnetBuilder.build_resnet_152_second_half_2(intermediate_shape_1, nb_classes)
# model_152_second_half_2.summary()

#
# model_152_second_half = resnet.ResnetBuilder.build_resnet_152_second_half(intermediate_shape, nb_classes)
# model_50_second_half = resnet.ResnetBuilder.build_resnet_50_second_half(intermediate_shape, nb_classes)
#
# model_50_second_half.load_weights("model/weights/50_second_half.h5")
# model_152_second_half.load_weights("model/weights/second_half_retrained.h5")

