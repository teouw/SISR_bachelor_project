import os
from model import SRCNN
from lib import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

scale = 3
c_dim = 3
dirname = './output/'
image_size = None


if __name__ == '__main__':

    #create the srcnn model
    srcnn = SRCNN(
        image_size=image_size,
        c_dim=c_dim,
        is_training=False)
    #load the testing data
    X_pre_test, X_test, Y_test = load_test(scale=scale, dim=c_dim)
    
    predicted_list = []
    for cnt, img in enumerate(X_test):

        start_time = time.time()
        if c_dim == 3:
            predicted = srcnn.process(img.reshape(1,img.shape[0],img.shape[1],3))
            predicted_reshaped = predicted.reshape(predicted.shape[1],predicted.shape[2],3)
        else:
            predicted = srcnn.process(img.reshape(1,img.shape[0],img.shape[1],1))
            predicted_reshaped = predicted.reshape(predicted.shape[1],predicted.shape[2],1)

        name = 'image{:02}'.format(cnt)  

        cv2.imwrite(os.path.join(dirname,name+'_input_image.bmp'), X_test[cnt])
        cv2.imwrite(os.path.join(dirname,name+'_original_image.bmp'), Y_test[cnt])
        cv2.imwrite(os.path.join(dirname,name+'_predicted_image.bmp'), predicted_reshaped)

