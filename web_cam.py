import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import cv2
import time

result_dir = 'results'

img_height, img_width = 150, 150
channels = 3
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.load_weights(os.path.join(result_dir, 'smallcnn.h5'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


cv2.namedWindow("Input")


cap = cv2.VideoCapture(0)
rval = True

while rval: 
        rval, frame = cap.read()
        if rval == False:
            break

        frame = cv2.resize(frame, (img_height, img_width))

        x = image.img_to_array(frame)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        pred = model.predict(x)[0]
        cv2.imshow("Input", frame)
        print(pred)
        
        key = cv2.waitKey(1)
        if key==27:
         break
cap.release()
cv2.destroyAllWindows()