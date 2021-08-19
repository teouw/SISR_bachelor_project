import os
import cv2
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

#Two simple function to load the training and testing data
def load_train(image_size=33, stride=33, scale=3, dim=3, load_txt=False):
    dirname = './data'
    dir_list = os.listdir(dirname)
    images = []

    #load the data
    images = [cv2.imread(os.path.join(dirname,img)) for img in dir_list if img != ".DS_Store"]
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]
    X_train = images.copy()
    Y_train = images.copy()

    #downsample and upsample to create the LR images
    X_train = [cv2.resize(img, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC) for img in X_train]
    X_train = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in X_train]

    sub_X_train  = []
    sub_Y_train = []

    #Creating the sub images of 33x33 
    for train, label in zip(X_train, Y_train):
        v = train.shape[0]
        h = train.shape[1]
        for x in range(0,v-image_size+1,stride):
            for y in range(0,h-image_size+1,stride):
                sub_train = train[x:x+image_size,y:y+image_size]
                sub_label = label[x:x+image_size,y:y+image_size]
                if dim == 3: 
                    sub_train = sub_train.reshape(image_size,image_size,3)
                    sub_label = sub_label.reshape(image_size,image_size,3)
                else:
                    sub_train = sub_train.reshape(image_size,image_size,1)
                    sub_label = sub_label.reshape(image_size,image_size,1)
                sub_X_train.append(sub_train)
                sub_Y_train.append(sub_label)
                # ========= VERIFICATION ===========
                # cv2.imshow('image',sub_train)
                # cv2.waitKey(0)

    #convert to numpy array
    sub_X_train = np.array(sub_X_train)
    sub_Y_train = np.array(sub_Y_train)

    return sub_X_train, sub_Y_train

def load_test(scale=3, dim=3):
    dirname = './input/'
    dir_list = os.listdir(dirname)
    #load the data
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.IMREAD_COLOR) for img in dir_list if img != ".DS_Store"]
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]
    X_test = images.copy()
    Y_test = images.copy()

    #downsample and upsample to create the LR images
    pre_tests = [cv2.resize(img, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC) for img in X_test]
    X_test = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in pre_tests]

    #reshape images to add the third channel
    if dim == 3: 
        pre_tests = [img.reshape(img.shape[0],img.shape[1],3) for img in pre_tests]
        X_test = [img.reshape(img.shape[0],img.shape[1],3) for img in X_test] 
        Y_test = [img.reshape(img.shape[0],img.shape[1],3) for img in Y_test] 
    else:
        pre_tests = [img.reshape(img.shape[0],img.shape[1],1) for img in pre_tests]
        X_test = [img.reshape(img.shape[0],img.shape[1],1) for img in X_test] 
        Y_test = [img.reshape(img.shape[0],img.shape[1],1) for img in Y_test]  

    return pre_tests, X_test, Y_test

