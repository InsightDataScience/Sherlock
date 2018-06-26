'''
Created on Jun 8, 2018

@author: runshengsong
'''
# TO DO
# move settings to config
import os
import sys
import json
import base64
import numpy as np

from collections import OrderedDict

def decode_pred_to_label(preds, model_name, num_return):
    """
    decode the prediction batch prediction results 
    to class label
    
    num_return: control the number of label to return
    """
    class_label = OrderedDict()
    json_file_path = os.path.join("app", "models", "InceptionV3", model_name, model_name + ".json")
    with open(json_file_path, 'r') as fp:
        class_label = json.load(fp)
    class_label = {int(k): str(v) for k,v in class_label.iteritems()}
    
    num_label = len(class_label)
    # sort the prob from high to low
    batch_output = []
    for each_image_pred in preds:
        this_image_result = []
        # get the index base on the prob from high to low
        this_classes = np.argsort(each_image_pred)[::-1]
        this_prob = np.sort(each_image_pred)[::-1]
        
        # trunct the list
        if num_return < num_label:
            this_classes = this_classes[:num_return]
            this_prob = this_prob[:num_return]

        # map classes number to label
        this_labels = map(class_label.get, this_classes)
        for i in range(0, len(this_labels)):
            one_lable = this_labels[i]
            one_prob = this_prob[i]
            this_image_result.append([one_lable, one_prob])
        
        # append the results of this image back to batch results
        batch_output.append(this_image_result)
    
    return batch_output
            
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
    