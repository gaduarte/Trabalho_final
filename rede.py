import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import matplotlib.pyplot as pt
import numpy as np

from keras.datasets import cifar10
batch_size = 20
epochs = 4
n_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

height = x_train.shape[1]
width = x_train.shape[2]

x_val = x_train[:2000,:,:,:]
y_val = y_train[:2000]
x_train = x_train[2000:,:,:,:]
y_train = y_train[2000:]

print('Train Dataset: ', x_train.shape, y_train.shape)
print('Validate dataset: ', x_val.shape, y_val.shape)
print('Test dataset: ', x_test.shape, y_test.shape)

image = x_train[4]

label = y_train
print(label)

y_train = np_utils.to_categorical(y_train, n_classes)
y_val = np_utils.to_categorical(y_val, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

print(y_train)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_val /= 255
x_test /= 255

def create_model():
  model = Sequential()
  model.add(Conv2D(filters = 80, kernel_size = (4,4), input_shape=(height, width,4), strides=1, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(filters = 80, kernel_size=(4,4), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(40, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))

  return model

  model = create_model()

  model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
  
  model.fit(x_train, y_train, batch_size = 20, epochs = 4, validation_data=(x_val, y_val), verbose=2)

  model.summary()
