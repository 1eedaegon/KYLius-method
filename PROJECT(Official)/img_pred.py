def number(arg1):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    im=Image.open(arg1)
    img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
    data = img.reshape([1, 784])
    data = 255 - data
    plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='nearest')
    print("MNIST predicted Number : ",sess.run(pred, feed_dict={X: data, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
    
# number("/home/itwill03/다운로드/numbers_image/numbers1.jpg")
