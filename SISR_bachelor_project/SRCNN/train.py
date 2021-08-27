from model import SRCNN
from lib import *
import sys

image_size= 33
c_dim=3
learning_rate = 0.003
batch_size = 32
epochs = 1000
stride = 14
scale = 4

image_shape = (image_size,image_size,3)

if __name__ == '__main__':
    try:
        srcnn = SRCNN(
            image_size=image_size,
            c_dim=c_dim,
            is_training=True,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,)
        X_train, Y_train = load_train(image_size=image_size, stride=stride, scale=scale, dim= c_dim)
        srcnn.train(X_train, Y_train)
    except:
        print("Please, verify that your dataset images match the correct dimension chosen.")