import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Cropping2D
import os
from math import ceil
import csv
import cv2
import sklearn

def generator(path, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size//2):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = path + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -center_angle
                images.append(center_image)
                images.append(image_flipped)
                angles.append(center_angle)
                angles.append(measurement_flipped)

            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)

samples = []
path = "/home/workspace/CarND-Behavioral-Cloning-P3/data/"
with open(path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[3] != "steering":
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Hyperparameters
batch_size= 128

# Dataset generators
train_generator = generator(path, train_samples, batch_size=batch_size)
validation_generator = generator(path, validation_samples, batch_size=batch_size)

# Model Architecture 
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5-1.0))
model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='elu'))
model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='elu'))
model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

#Compile
model.compile(loss='mse', optimizer='adam')

# Load prev model
# model.load_weights('dt.h5')

# Train & validate
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)

# Save model
model.save("final.h5")
