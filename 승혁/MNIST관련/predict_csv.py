def load_optimizer(opt_addr, opt_addr2, data):
    import tensorflow as tf
    # initialize/ load
    global saver=tf.train.import_meta_graph(opt_addr+".meta")
    global sess = tf.InteractiveSession()
    print("Meta_Graph Imported")
    
    saver.restore(sess, tf.train.get_checkpoint_state(opt_addr2).model_checkpoint_path)
    print("Parameters Restored")
    
    global graph=tf.get_default_graph()
    global X=graph.get_tensor_by_name('X:0')
    global pred=graph.get_tensor_by_name('pred:0')
    global p_keep_conv=graph.get_tensor_by_name('p_keep_conv:0')
    global p_keep_hidden=graph.get_tensor_by_name('p_keep_hidden:0')
    print("Variables Saved")
