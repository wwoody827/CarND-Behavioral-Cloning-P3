import cv2
import csv
import numpy as np
import random, math
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if len(batch_sample) < 4:
                    continue
                filename = batch_sample[0].replace('\\', '/')
                name = './data/IMG/'+filename.split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                # if abs(center_angle) < 0.05:
                #    if random.random() > 0.5:
                #        continue
                
                images.append(center_image)
                angles.append(center_angle)
                
                fliped_image = np.fliplr(center_image)
                images.append(fliped_image)
                angles.append(-center_angle)
                
                filename = batch_sample[1].replace('\\', '/')
                name = './data/IMG/'+filename.split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3]) + 0.25
                images.append(center_image)
                angles.append(center_angle)
                
                fliped_image = np.fliplr(center_image)
                images.append(fliped_image)
                angles.append(-center_angle)

                filename = batch_sample[2].replace('\\', '/')
                name = './data/IMG/'+filename.split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3]) - 0.25
                images.append(center_image)
                angles.append(center_angle)
                
                fliped_image = np.fliplr(center_image)
                images.append(fliped_image)
                angles.append(-center_angle)
   
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
samples = []
csv_filename = './data/driving_log.csv'
with open(csv_filename, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        samples.append(line)
        
samples = samples[1:]
random.shuffle(samples)
train_size = math.floor(0.8 * len(samples))
train_samples = samples[0:train_size] 
validation_samples = samples[train_size+1:]

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


ch, row, col = 3, 160, 320  # Trimmed image format


# model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers


# nvidia model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50,10), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(3,1,1,border_mode='valid', activation='relu', subsample=(1,1), W_regularizer = regularizers.l2(0.001)))

model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizers.l2(0.001)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizers.l2(0.001)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(2,2), W_regularizer = regularizers.l2(0.001)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1164, activation='relu', W_regularizer = regularizers.l2(0.001)))
model.add(Dense(100, activation='relu', W_regularizer = regularizers.l2(0.001)))
model.add(Dense(50, activation='relu',  W_regularizer = regularizers.l2(0.001)))
model.add(Dense(10, activation='relu',  W_regularizer = regularizers.l2(0.001)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(train_generator, samples_per_epoch = 4 * len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')



