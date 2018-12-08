from PIL import Image
import glob
import os
#from resizeimage import resizeimage
#from keras import backend as K
#from six.moves import cPickle
#from keras.datasets import cifar10
import numpy as np
import csv
import codecs
import random
from skimage.transform import resize
from skimage import io
import re

image_size = 224

r1 = re.compile("frame_(\d+)")
r2 = re.compile("patch_(\d+)")
def key1(a):
    m1 = r1.findall(a)
    m2 = r2.findall(a)
    return int(m1[0]), int(m2[0])

#resize images
def resize_Images():
    for filepath in glob.iglob('data/original/*.jpg'):
        filename = os.path.basename(filepath)

        image = io.imread(filepath)
        resized = resize(image, (image_size, image_size), anti_aliasing=True)        
        io.imsave('dataset/resized224/' + filename, resized)

        #with open(filepath, 'r+b') as f:
            #with Image.open(f) as image:
                #resized = resizeimage.resize_crop(image, [image_size, image_size])
                #resized.save('dataset/resized224/' + filename, image.format)
        

def load_Data():
    num_train_samples = 2273
    testset_size = 450

    x_train = np.empty((num_train_samples, image_size, image_size, 3), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    (all_x, all_y) = load_batch()

    random_choice = random.sample(range(num_train_samples), testset_size)
    rest = np.setdiff1d(range(num_train_samples), random_choice)
  
    y_test = np.take(all_y, random_choice)
    y_train = np.take(all_y, rest)
    
    x_train = []
    x_test = []
    for index in range(num_train_samples):
        if index in random_choice:
            x_test.append(all_x[index])
        else:
            x_train.append(all_x[index])
	
    x_train = np.array(x_train)
    x_test = np.array(x_test)	

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    return (x_train, y_train), (x_test, y_test)

def load_batch():
    #image data
    imageCount = 0
    filelist = []
    for imagefile in glob.iglob('data/resized224/*.jpg'):
        filelist.append(imagefile)
        imageCount += 1

    filelist.sort(key = key1)
    data = np.zeros((imageCount, image_size, image_size, 3))
    labels = np.zeros(imageCount)
    index = 0

    for counter, imagefile in enumerate(filelist):
        print(str(counter + 1) + 'load image file: ' + imagefile)
        t = Image.open(imagefile)
        arr = np.array(t) #Convert test image into an array 32*32*3    
        data[index] = arr 
        index += 1
        
    #labels
    labels = []
    #for filepath in glob.iglob(''):
    with open('data/labels/all.txt') as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            labels.append(line.strip().split(',')[1])    
    return data, labels

<<<<<<< HEAD:Reset/data_prepare.py
# load_Data()
=======
load_Data()
>>>>>>> 663d13dd08abe400d1b4df1691f6eb734834c3a5:oldResnet/data_prepare.py