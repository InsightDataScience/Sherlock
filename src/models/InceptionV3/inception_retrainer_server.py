'''
Created on Jun 12, 2018

@author: runshengsong
'''
import redis
import time
import json
import numpy as np
from rq import Queue

# inception helpers
import helpers
import settings

# Keras
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils

db = redis.StrictRedis(host="localhost", port=6379,
                       db=0)


def run_inceptionV3_retrainer_server():
    '''
    run the retraining server for Inception V3
    '''
    pass