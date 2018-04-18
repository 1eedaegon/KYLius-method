#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 00:50:27 2018

@author: kimseunghyuck
"""
import sys
sys.path

from PIL import Image
import numpy
img=Image.open("/Users/kimseunghyuck/desktop/number1.jpg")
imgarray=numpy.array(img)
