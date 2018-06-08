'''
Created on Jun 7, 2018

@author: runshengsong
'''
import os

import numpy as np
from scipy.misc import imread
from keras.utils import np_utils
from keras.models import load_model

#TO DO
# move model path to configuration
MNIST_PATH = os.path.join("app", "models", "mnist", "current", "mnist_baseline_less.h5")

class mnist:
    def __init__(self):
        pass
    
    def fit(self, img):
        '''
        fit the existing model using the new input data
        '''
        # TO DO
        img_to_add = imread()
        
    
    def predict(self, img_input):
        """
        @img_input: raw image files
        
        @return: the predicted number of img_input
        """
        # preprocess
        x = imread(img_input, mode = 'L')
        x = np.invert(x).reshape(1, 784).astype('float32') / 255

        model = load_model(MNIST_PATH)
        
        out = model.predict(x)
        return np.argmax(out)
    
    def pre_process(self, imgs, label):
        """
        format the images 
        """
        # flatten 28*28 images to a 784 vector for each image
        # deal with a single img
        imgs = imgs.reshape(imgs.shape[0], 1, 28, 28).astype('float32')
        
        # normalization
        imgs = imgs / 255
        
        # flat the label
        label = np_utils.to_categorical(label, num_classes=10)
        
        return img, label
        