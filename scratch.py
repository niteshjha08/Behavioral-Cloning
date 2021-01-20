import memory_profiler
import pandas as pd
import csv
import os
m1=memory_profiler.memory_usage()
import cv2
images=[]
# with open("driving_log.csv") as f:
#     reader=csv.reader(f)
#     for line in reader:
#         print(line)
#         image=cv2.imread(line[0])
#         cv2.imshow('image:'.format(line),image)
#         images.append(image)
#
# m2=memory_profiler.memory_usage()
# print(m1[0]-m2[0])
# def image_load():
#     with open("driving_log.csv") as f:
#         reader=csv.reader(f)
#         for line in reader:
#             print(line)
#             image=cv2.imread(line[0])
#             yield(image)
# images=image_load()
# for dir,subdir,files in os.walk('./samples'):
#     images=files
#
# for image in images:
#     print(image)
#     img = cv2.imread('samples/' + image)
#     print(img.shape)
#     cv2.imshow('this is it',img)
#     cropped= img[60:140,:,:]
#     cv2.imshow('cropped', cropped)
#     cv2.waitKey(0)
# m2=memory_profiler.memory_usage()
# for i in images:
#     cv2.imshow('image:',i)
#     cv2.waitKey(10)
# print(m1[0]-m2[0])
data_path=''
lines=[]
# with open(data_path + 'driving_log.csv') as f:
#     reader = csv.reader(f)
#     next(reader, None)
#     for line in reader:
#         lines.append(line)
img=cv2.imread('images/center.PNG')
flipped=cv2.flip(img,1)
cv2.imshow('original',img)
cv2.imshow('flipped',flipped)
cv2.waitKey()
# print(lines.)