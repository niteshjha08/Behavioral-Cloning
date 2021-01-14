import cv2
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

lines=[]
i=0
with open("driving_log.csv") as f:
    reader=csv.reader(f)
    for line in reader:
        lines.append(line)

images=[]
steering=[]
total_images=len(lines)

for i in range(total_images):
    path='./IMG/'+lines[i][0].split('\\')[-1]
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(image)

    steer_val=lines[i][3]
    steering.append(steer_val)


images=np.array(images)
print("Shape of images input is:",images.shape)
steering=np.array(steering)
print("Shape of steering input is:", steering.shape)

images=images/255.0 - 0.5
model=keras.Sequential()
model.add(Conv2D(10,3,padding='valid',input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(images,steering,validation_split=0.2,epochs=5)

model.save('model.h5')


