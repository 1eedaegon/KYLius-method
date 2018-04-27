# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:05:50 2018

@author: stu
"""
import numpy as np
import pandas as pd

td = pd.read_csv("c:/data/sound/feature_train.csv",delimiter=',')
labels = train_info['label']
l = train_info['label'].unique()

df_label = pd.DataFrame(labels)

for i in range(len(l)):
    df_label[df_label==l[i]] = int(i)
df_label.values
   
df_label.to_csv("c:/data/sound/audio_label.csv",index=False,mode='w')
td = pd.read_csv("c:/data/sound/audio_label.csv",delimiter=',')
