#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:50:27 2018

@author: kimseunghyuck
"""
def img_to_csv(imgaddr):
    from PIL import Image
    import numpy
    #img=Image.open("/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/승혁/numbers_image/numbers1.jpg")
    img=Image.open(imgaddr)
    size=(28,28)
    #사진을 흑백으로 바꿈
    im2=img.convert("L")
    #사진 사이즈를 28*28로 바꿈.
    im2=im2.resize(size)
    #csv로 변환
    imgarray=numpy.array(im2)
    return imgarray
    
