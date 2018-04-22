# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 01:19:53 2018

@author: modes
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/data/data3.png')
#img = cv2.IMREAD_GRAYSCALE('/data/data3.png') #GRAYSCALE로 읽음 행과열만 리턴

#color problem-solving
    #way1 : image channels change BGR to RBG
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

img[:,:,0] = r
img[:,:,1] = b
img[:,:,2] = g
    #way2 : convert color space 
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#image exploing
img.shape # original : (293, 293, 3), converted image to grayscale: (293, 293)
img[0].shape
img.dtype

"""
Kaggle data는 grayscale로 convert한 데이터 이므로 way2를 사용한다
"""

#matplot 으로 출력
fig, axis = plt.subplots(1,1, figsize=(5,5))
axis.imshow(img)

#opencv 으로 출력
cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

"""
#높이 넓이 나누기
height, width = img.shape[:2]
"""

"""
h1=img[0][0:6]
h2=img[1][0:6]
h3=img[2][0:6]

a=np.array([h,w])

cv2.imshow('a',a)
cv2.waitkey()

cv2.destroyAllWindows()
"""
#pixel values
px = img[100,200]
px

    #np이용 color값 바꾸기
img.item(10,10,2)
img.itemset((10,10,2),100)
img.item(10,10,2)


#Image ROI : Region of Image(ROI)
    #image slicing
w1_im = img[0:100,0:100] #0~100행 0~100열
w1_im.shape

    #image printing _ opencv 
cv2.imshow('w1_im',w1_im)
cv2.waitKey(0)

cv2.destroyAllWindows()


    #image printing _ maptplot
fig, axis = plt.subplots(1,1, figsize=(5,5))
axis.imshow(w1_im)

#Image ROI _ strided by 10,10,10,10
    #image slicing
w2_im = img[10:110,10:110] #10~110행 10~110열

    #image printing _ opencv 
cv2.imshow('w2_im',w2_im)
cv2.waitKey()

cv2.destroyAllWindows()


    #image printing _ matplotlib
fig, axis = plt.subplots(1,1, figsize=(5,5))     
axis.imshow(w2_im)


#image printing operated on weight values 
w1 = 3
a = w2_im * w1

fig, axis = plt.subplots(1,1, figsize=(5,5))
axis.imshow(a)