import cv2
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
import os
lines=[]
i=0
data_path="./../../opt/carnd_p3/data/"

with open(data_path + 'driving_log.csv') as f:
    reader=csv.reader(f)
    next(reader,None)
    for line in reader:
        lines.append(line)

images=[]
steering=[]
total_images=len(lines)

for i in range(total_images):
    print(i)
    path_center= data_path + 'IMG/' + lines[i][0].split('/')[-1]
    image_center=cv2.imread(path_center)
    image_center=cv2.cvtColor(image_center,cv2.COLOR_BGR2RGB)
    image_center=image_center/255.0 - 0.5
    images.append(image_center)
    steer_val= np.float(lines[i][3])
    steering.append(steer_val)

    path_left = data_path + 'IMG/' + lines[i][1].split('/')[-1]
    image_left = cv2.imread(path_left)
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_left=image_left/255.0 - 0.5
    images.append(image_left)
    steer_val = np.float(lines[i][3]) + 0.3
    steering.append(steer_val)

    path_right = data_path + 'IMG/' + lines[i][2].split('/')[-1]
    image_right = cv2.imread(path_right)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    image_right=image_right/255.0 - 0.5
    images.append(image_right)
    steer_val = np.float(lines[i][3]) - 0.3
    steering.append(steer_val)

images=np.array(images)
print("Shape of images input is:",images.shape)
steering=np.array(steering)
print("Shape of steering input is:", steering.shape)

#images=images/255.0 - 0.5

images,steering=shuffle(images,steering)
print("Model is being made ...")
model=keras.Sequential()
model.add(Conv2D(20,3,padding='valid',input_shape=(160,320,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Conv2D(40,3,padding='valid'))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
print("Model completed")
#print(model.summary())

model.compile(loss='mse',optimizer='adam')
model.fit(images,steering,validation_split=0.2,epochs=5)
model.save('model_sample_data.h5')
