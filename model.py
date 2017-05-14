import cv2
import csv
import numpy as np

# read data
lines = []
csv_filename = './data/driving_log.csv'
with open(csv_filename, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)


images = []
measurements = []

for line in lines[1:]:
    # print(line)
    image_path = './data/'+ line[0]
    image = cv2.imread(image_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)
    

    
# print(image_path)
images = np.array(images)
measurements = np.array(measurements)

X_train = images
y_train = measurements



# data process


# model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)
model.save('model.h5')



