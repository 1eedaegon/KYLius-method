#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:37:40 2018

@author: kimseunghyuck
"""
#addr(경로, 파일이름 포함) 파일의 이름 뒤에 x를 덧붙임.
def file_rename(addr, x):
    import os
    #fname = "/Users/kimseunghyuck/desktop/git/daegon/KYLius-method/승혁/numbers_image/numbers2.jpg"
    fname=addr
    frename=os.path.splitext(fname)[0]+str(x)+os.path.splitext(fname)[-1]
    os.rename(fname, frename)
    print("파일이름이 {}에서 {}로 변경되었습니다".format(fname, frename))
    