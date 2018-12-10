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




# build and load weights for resnet 50
model_50 = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model_50.load_weights("model/weights/50weights.h5")

#build and load weights for resnet 152
model_152 = resnet.ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)
model_152.load_weights("model/weights/152weights.h5")

# set intermediate input shape and build second half model for reset 152
intermediate_shape = (28, 28, 512)
output_layer = 74
input_layer = 75

model_152_second_half = resnet.ResnetBuilder.build_resnet_152_second_half_2(intermediate_shape, nb_classes)
# model_50_second_half = resnet.ResnetBuilder.build_resnet_50_second_half(intermediate_shape, nb_classes)

#method for load weights into sec_half model
def batch_set_weights(model1, n_layer1, model2, n_layer2):
    for i, j in zip(range(n_layer1, len(model1.layers)),range(n_layer2, len(model2.layers))):
        model1.layers[i].set_weights(model2.layers[j].get_weights())
        print(i,j)

# batch_set_weights(model_50_second_half, 1, model_50, input_layer)
# model_50_second_half.save_weights("model/weights/50_second_half.h5")

batch_set_weights(model_152_second_half, 1, model_152, input_layer)
model_152_second_half.save("model/model/second_half.h5")

# model_50_second_half.load_weights("model/weights/50_second_half.h5")
# model_152_second_half.load_weights("model/weights/second_half_retrained.h5")


intermediate_layer_model = Model(inputs=model_50.input,
                                 outputs=model_50.layers[output_layer].output)

feature_map_train = intermediate_layer_model.predict(X_train)
feature_map_test = intermediate_layer_model.predict(X_test)

# output_path = "model/feature_map/"
# np.save(output_path + "feature_map_train.npy", feature_map_train)
# np.save(output_path + "feature_map_test.npy", feature_map_test)
# np.savez_compressed(output_path + "feature_map_train.npy", feature_map_train)
# np.savez_compressed(output_path + "feature_map_test.npy", feature_map_test)

def batch_compile(models):
    for model in models:
        model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

# batch_compile([model_50, model_152, model_50_second_half, model_152_second_half])

# train second half model using feature_map_train
# model_second_half.fit(feature_map_train, Y_train,
#           batch_size=batch_size,
#           nb_epoch=nb_epoch,
#           validation_data=(feature_map_test, Y_test),
#           shuffle=True,
#           callbacks=[lr_reducer, early_stopper, csv_logger])

# model_152_second_half.save_weights("model/weights/second_half_retrained.h5")
# loss, acc = model_152_second_half.evaluate(feature_map_test, Y_test)

n_inference = 5
# print("------------ TEST 1: resnet 50 + 152 -------------")
# start_time = time.time()
# for i in range(n_inference):
#     print("ROUND %s:" % (i))
#     round_start = time.time()
#     model_50.predict(X_test)
#     curr_time = time.time()
#     print("resnet 50 finished in % seconds" % (curr_time - round_start))
#     curr_time = time.time()
#     model_152.predict(X_test)
#     print("resnet 152 finished in % seconds" % (time.time() - curr_time))
#     print("round total time: % seconds" % (time.time() - round_start))
# time1 = time.time() - start_time
# print("------------ %s rounds of inference took %s seconds ------------" % (n_inference, time1))

#
# print("------------ TEST 2: feature map + partial resnet 50 + 152 -------------")
# start_time = time.time()
# for i in range(n_inference):
#     print("ROUND %s:" % (i))
#     round_start = time.time()
#     feature_map_test = intermediate_layer_model.predict(X_test)
#     curr_time = time.time()
#     print("feature map finished in % seconds" % (curr_time - round_start))
#     curr_time = time.time()
#     model_50_second_half.predict(feature_map_test)
#     print("partial 50 finished in % seconds" % (time.time() - curr_time))
#     curr_time = time.time()
#     model_152_second_half.predict(feature_map_test)
#     print("partial 50 finished in % seconds" % (time.time() - curr_time))
#     print("round total time: %s seconds" % (time.time() - round_start))
# time2 = time.time() - start_time
# print("------------ %s rounds of inference took %s seconds ------------" % (n_inference, time2))

# print("time diff: %s" % (time1 - time2))

# pred = model_second_half.predict(feature_map)
#
# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               validation_data=(X_test, Y_test),
#               shuffle=True,
#               callbacks=[lr_reducer, early_stopper, csv_logger])
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
#
#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(X_train)
#
#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                         steps_per_epoch=X_train.shape[0] // batch_size,
#                         validation_data=(X_test, Y_test),
#                         epochs=nb_epoch, verbose=1, max_q_size=100,
#                         callbacks=[lr_reducer, early_stopper, csv_logger])


print("training finished")
