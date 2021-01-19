import cv2
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from math import ceil

lines = []
data_path = "./../../opt/carnd_p3/data/"


# data_path=""
def generator(filepaths, batch_size):
    n_samples = len(filepaths)
    while True:
        shuffle(filepaths)

        for i in range(0, n_samples, batch_size):
            batch = filepaths[i:i + batch_size]
            images = []
            steer_angles = []
            for sample in batch:
                path = data_path + 'IMG/' + sample[0]
                # print(path)
                image = cv2.imread(path)
                image = preprocess_images(image)
                images.append(image)
                steer_angles.append(np.float(sample[1]))

            images = np.array(images)
            steer_angles = np.array(steer_angles)

            yield shuffle(images, steer_angles)


def image_loader(path):
    images = []
    steering_angle = []
    for line in path:
        image = cv2.imread('IMG/' + line[0].split('\\')[-1])
        image = preprocess_images(image)
        images.append(image)
        steering_angle.append(line[3])
    images = np.array(images)
    steering_angle = np.array(steering_angle)
    shuffle(images, steering_angle)
    return (images, steering_angle)


def preprocess_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:140, :, :]
    image = image / 255.0 - 0.5
    return image


with open(data_path + 'driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        lines.append(line)

all_data = []
steering_correction = 0.3
for i in range(len(lines)):
    all_data.append([lines[i][0].split('/')[-1], np.float(lines[i][3])])
    all_data.append([lines[i][1].split('/')[-1], np.float(lines[i][3]) + steering_correction])
    all_data.append([lines[i][2].split('/')[-1], np.float(lines[i][3]) - steering_correction])
train_lines, validation_lines = train_test_split(all_data, test_size=0.2)

BATCH_SIZE = 32

train_generator = generator(train_lines, BATCH_SIZE)
validation_generator = generator(validation_lines, BATCH_SIZE)

model = keras.Sequential()
model.add(Conv2D(24, 5, padding='valid', input_shape=(80, 320, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(36, 5, padding='valid'))
model.add(Conv2D(48, 5, padding='valid'))
model.add(Conv2D(64, 3, padding='valid'))
model.add(Conv2D(64, 3, padding='valid'))

model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.5))
model.add(Dense(100))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print("Model completed")
# print(model.summary())

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
                    steps_per_epoch=ceil(len(train_lines) / BATCH_SIZE), \
                    validation_data=validation_generator, \
                    validation_steps=ceil(len(validation_lines) / BATCH_SIZE), \
                    epochs=5, verbose=1)
# model.fit(images,angles,validation_split=0.2, epochs=5)
model.save('model.h5')
