import csv
import numpy as np
import tensorflow as tf 
import keras
import cv2
from keras import regularizers
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.utils import np_utils
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Lambda, Cropping2D



lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images =[]
measurements = []
correction = 0.13

for line in lines:
    # for center images
    source_path = line[0]    
    filename = source_path.split('/')[-1]
    current_path = 'IMG/' + filename     # path of center image
    image= cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])     #steering angles for images from center camera
    measurements.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)
    
    # for left images
    source_path_l = line[1]    
    filename_l = source_path_l.split('/')[-1]
    current_path_l = 'IMG/' + filename_l     # path of left image
    image_l= cv2.cvtColor(cv2.imread(current_path_l),cv2.COLOR_BGR2RGB)
    images.append(image_l)
    measurement_l = float(line[3]) + correction     #steering angles correction for images from left camera
    measurements.append(measurement_l)
    image_flipped_l = np.fliplr(image_l)
    measurement_flipped_l = -measurement_l
    images.append(image_flipped_l)
    measurements.append(measurement_flipped_l)
    
    # for right images
    source_path_r = line[2]    
    filename_r = source_path_r.split('/')[-1]
    current_path_r = 'IMG/' + filename_r    # path of right image
    image_r= cv2.cvtColor(cv2.imread(current_path_r),cv2.COLOR_BGR2RGB)
    images.append(image_r)
    measurement_r = float(line[3]) - correction     #steering angles correction for images from right camera
    measurements.append(measurement_r)
    image_flipped_r = np.fliplr(image_r)
    measurement_flipped_r = -measurement_r
    images.append(image_flipped_r)
    measurements.append(measurement_flipped_r)
  
x_train = np.array(images)
y_train = np.array(measurements)
print(len(x_train))

model = models.Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))   #normalization

model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))   #crop out the part of sky and engine block.

model.add(layers.Conv2D(24, (5, 5))) 
model.add(Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(36, (5, 5)))
model.add(Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(48, (5, 5)))
model.add(Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3))) 
model.add(Activation('relu'))

model.add(layers.Conv2D(64, (3, 3))) 
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(layers.Flatten()) 
model.add(layers.Dense(100)) 
model.add(layers.Dense(50))
model.add(layers.Dense(10))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split = 0.2, shuffle = True)    #training/validating
model.save('modet.h5')    