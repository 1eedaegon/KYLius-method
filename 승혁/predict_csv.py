#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 01:33:49 2018

@author: kimseunghyuck
"""
def load_optimizer(opt_addr, opt_addr2, data):
    import tensorflow as tf
    import pandas as pd
    """
    #path="/home/paperspace/Downloads/"
    #path2="desktop/git/daegon/KYLius-method/승혁/"
    #데이터 가져오기
    train = pd.read_csv("desktop/git/daegon/KYLius-method/승혁/train.csv")
    #train = pd.read_csv('/home/itwill03/다운로드/train.csv')
    
    #훈련세트, validation세트 나누기(여기서는 validate만 필요)
    from sklearn.model_selection import train_test_split
    train_set, validate_set = train_test_split(train, test_size = 0.3)
    #trainData = train_set.values[:,1:]
    validateData1 = validate_set.values[:,1:].astype(float)
    #trainLabel=train_set.values[:,0]
    validateLabel1 = validate_set.values[:,0]
    """
    # initialize/ load
    saver=tf.train.import_meta_graph(opt_addr+".meta")
    sess = tf.InteractiveSession()
    print("Meta_Graph Imported")
    saver.restore(sess, tf.train.get_checkpoint_state(opt_addr2).model_checkpoint_path)
    print("Parameters Restored")
    
    graph=tf.get_default_graph()
    X=graph.get_tensor_by_name('X:0')
    pred=graph.get_tensor_by_name('pred:0')
    p_keep_conv=graph.get_tensor_by_name('p_keep_conv:0')
    p_keep_hidden=graph.get_tensor_by_name('p_keep_hidden:0')
    print("Variables Saved")
    # test the data
    print(sess.run(pred, feed_dict={X: data, 
                    p_keep_conv: 1.0, p_keep_hidden: 1.0}))
    sess.close()