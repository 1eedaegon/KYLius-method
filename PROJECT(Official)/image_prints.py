# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:07:50 2018

@author: stu
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_data = cv2.imread('/data/data3.png')
image_data = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#single image printing function
    #im_num : number of image , size : output_size 
def sing_prt(im_num, size):
    fig1, axis = plt.subplots(1,1, figsize=(size,size))
    axis.imshow(img[0].reshape(28,28))

sing_prt(5,2)

#multi images printing function 
    #im_num : number of image , size : output_size 
def mul_prt(num,size):
    fig1, axis = plt.subplots(1,num, figsize=(size,size))
    for i in range(num):
        axis[i].imshow(img[i].reshape((28,28)))
        axis[i].axis('off')
        axis[i].set_title(label[i])
        
mul_prt(5,8)

# convolutional images printing function
    # start,end : 이미지 행렬 좌표의 시작점, 끝점
    # strides : striders
    # num : number of output
    # size : output size
def conv_prt(start,end,strides,num,size):
    image = image_data
    image = image[start:end,start:end]
    
    st = start
    en = end
    number = num
        
    fig1, axis = plt.subplots(1,number, figsize=(size,size))
    for i in range(number):
        axis[i].imshow(image[st:en,st:en])
        axis[i].axis('off')
        axis[i].set_title(label[i])
        st += strides
        en += strides
        
conv_prt(0,100,5,5,10)    

#images printing function (operated on weight values) 
    # start,end : 이미지 행렬 좌표의 시작점, 끝점
    # w : weight 개수
    # size : output size
def conv_prt(start,end,w,size):
    image = image_data
    image = image[start:end,start:end]
    
    st = start
    en = end
        
    fig1, axis = plt.subplots(1,number, figsize=(size,size))
    for i in range(w):
        axis[i].imshow(image[st:en,st:en]*(i+1))
        axis[i].axis('off')
        axis[i].set_title(label[i])

        
conv_prt(0,100,5,5,10)    



