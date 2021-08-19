from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
import argparse
from lib import *
from model import *

input_dir = './input/'
output_dir = './output/'
model_name = './model/srgan_model.h5'
image_shape = (96,96,3)
scale = 2


if __name__== "__main__":

    loss = VGG_LOSS(image_shape)  
    model = load_model(model_name , custom_objects={'vgg_loss': loss.vgg_loss})
    
    x_test_lr, x_test_hr, pre_input = process_testing_data(input_dir, scale)
    plot(output_dir, model, x_test_hr, x_test_lr, pre_input, scale)




        





