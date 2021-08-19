from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import cv2
import time

##############################################################################
#This function is used to load the data in 3 channel images.
#In addtion we make sure that the shape size dimension is in good shape to be downsample
#We do the modulo of the shape with the scale number that we will use for the downsample and we remove the extra pixels
#to the original image.
#Finally, we return the dataset.
def load_data(dirs, scale=3):
    files = []
    dir_list = os.listdir(dirs)
    files = [cv2.cvtColor(cv2.imread(os.path.join(dirs,img)),cv2.IMREAD_COLOR) for img in dir_list if img != ".DS_Store"]
    files = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in files]

    return files 

##############################################################################
#In order for the model to process thhe data, it needs to be normalize.
#that means, we need to add a new batch size axis and we need to convert it
# from uint8 to float32. Then, each values will be divided by 255 as needed.
def normalize(input_data):
    #Add the batch size axis
    x_train = np.expand_dims(input_data, axis=-1) 
    #Convert it to model's acceptable data type
    x_train = x_train.astype('float32') / 255
    return x_train
#The model output images are still in float with value in a range not displayable.
#So we have to change the values to a uint8 range(between 0 and 255). For that we add 1 to each value
# and we multiple by 255/2. We do that because the model last activation function is "tanh".
# and then convert it back to uint8 shape and return it
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

##############################################################################
#We need a function that will downsample a HR image to a LR image.
#We can use the resize function implemented by OPENCV2 to interpolate an image
#with a method and a scale chosen.
#The interpolation used is "Bicubic"
#we then return the new LR dataset
def hr_to_lr(images_real , scale):
    images = []
    for img in  range(len(images_real)):
        images.append(cv2.resize(images_real[img], None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC))
    images_lr = np.array(images)
    return images_lr

##############################################################################
#We need a function that will process the data to test the model.
#To do that, we first load the data using the directory name given in parameter.
#Then, from the original SR dataset we downsampled the images by a factor given in parameter. 
#These images will be the model's LR input. So we have to normalize it. We return the LR input
#as well as the original images and the input images before the normalization for display purpose.
def process_testing_data(directory, scale=3):  
    files = load_data(directory, scale)
    #convert to numpy array
    original_images = array(files)
    pre_lr_input = files.copy()
    #reshape to fit the model acceptable input dimension
    pre_lr_input = [cv2.resize(img, (96,96), interpolation=cv2.INTER_CUBIC) for img in pre_lr_input] 
    #downsample the original dataset to LR images by a factor of "scale"
    lr_images = hr_to_lr(files, scale)
    #reshape to fit the model acceptable input dimension
    lr_images = [cv2.resize(img, (96,96), interpolation=cv2.INTER_CUBIC) for img in lr_images]
    #we normalize the LR input for the model
    lr_images = normalize(lr_images)
    return lr_images, original_images, pre_lr_input

##############################################################################
#We need a function that will process the data to train the model.
#To do that, we first load the data using the directory name given in parameter.
#We then split the original dataset into a training and a testing dataset. The numbers of
#images on both datasets will be determined by the split ratio. 
# ex: if we have 10 images and the ratio is at 0.8 -> 8 of 10 images will be in the training dataset.
#In addition, we split both datasets into LR images and HR images.
#these four datasets will be normalized to fit into the model.
#both LR datasets are downsampled before normalized and both HR datasets are convert to numpy array.
#we finally return them.
def process_training_data(directory, split_ratio = 0.8, scale=3):
    #load the dataset
    files = load_data(directory, scale)
    #getting the number of images for the training dataset
    number_of_train_images = int(len(files) * split_ratio)
    #split the original dataset into training and testing
    training_dataset = files[:number_of_train_images]
    testing_dataset = files[number_of_train_images:len(files)]
    #convert then normalize
    x_train_hr = array(training_dataset)
    x_train_hr = normalize(x_train_hr)
    #downsample then normalize
    x_train_lr = hr_to_lr(training_dataset, scale)
    x_train_lr = normalize(x_train_lr)
    #convert then normalize
    x_test_hr = array(testing_dataset)
    x_test_hr = normalize(x_test_hr)
    #downsample then normalize
    x_test_lr = hr_to_lr(testing_dataset, scale)
    x_test_lr = normalize(x_test_lr)
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr

##############################################################################
#To compare the model perfomance,i used those three metrics.
#The formulas were taken from:
#MSE: https://en.wikipedia.org/wiki/Mean_squared_error
#PSNR: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#SSIM: https://fr.wikipedia.org/wiki/Structural_Similarity
def mse(y, t):
    return np.mean(np.square(y - t))

def psnr(y, t):
    return 20 * np.log10(255) - 10 * np.log10(mse(y, t))

def ssim(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mu_x) * (y - mu_y))
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))

##############################################################################
# This function is used to plot the HR image, LR input image and, the predicted HR image
# I then save the plot into the "result" folder.
def plot(output_dir, generator, hr_images, lr_input_images , pre_input, scale=3, dim=(1, 3), figsize=(15, 5)):
   
    nbr_of_images = hr_images.shape[0]
    #predict
    gen_img = generator.predict(lr_input_images)

    for index in range(nbr_of_images):
        #create plot
        fig = plt.figure(figsize=figsize)
    
        ax1 = plt.subplot(dim[0], dim[1], 1)
        plt.imshow(cv2.cvtColor(hr_images[index].squeeze(), cv2.COLOR_BGR2RGB))
        ax1.set_title('High resolution image')
        plt.axis('off')

        ax2 = plt.subplot(dim[0], dim[1], 2)
        plt.imshow(cv2.cvtColor(lr_input_images[index].squeeze(), cv2.COLOR_BGR2RGB))
        ax2.set_title('Low resolution image')
        plt.axis('off')

        ax3 = plt.subplot(dim[0], dim[1], 3)
        plt.imshow(cv2.cvtColor(gen_img[index].squeeze(), cv2.COLOR_BGR2RGB))
        ax3.set_title('Predicted image')
        plt.axis('off')
    
        #save the plot
        fig.savefig(output_dir + 'bsds100_%d.png' % index)


