'''
Created on Jun 8, 2018

@author: runshengsong
'''
# TO DO
# move settings to config
import sys
import base64
import numpy as np

def pre_process_image(img):
    """
    format the images 
    """
    # To a vector
    x = np.array([np.array(img)])
    # flatten 28*28 images to a 784 vector for each image
    # deal with a single img
    x = x.reshape(x.shape[0], 1, 28, 28).astype('float32')
    
    # normalization
    x = x / 255
    
    return x

def base64_encode_image(a):
    """
    encode the image
    """
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
    """
    decode the image
    """
    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a
    