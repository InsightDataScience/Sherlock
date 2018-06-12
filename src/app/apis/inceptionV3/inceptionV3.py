'''
Created on Jun 11, 2018

Web service for InceptionV3

@author: runshengsong
'''

from PIL import Image
import json
import numpy as np
import uuid
import os
import io
import uuid
import time

# keras
from keras.models import load_model
from keras.preprocessing import image
from keras_applications import inception_v3

# flask
from flask import jsonify
from flask import Blueprint, request

import helpers

from app import app
from app import db


blueprint = Blueprint('inceptionV3', __name__)

INCEPTIONV3_IMAGE_QUEUE = app.config['INCEPTIONV3_IMAGE_QUEUE']
CLIENT_SLEEP = app.config['CLIENT_SLEEP']

@blueprint.route('/predict', methods=['POST'])
def run_inceptionV3():
    """
    Listening user submitted images and 
    stack them in a Redis queue
    """
    data = {"success": False}
    
    # load image
    img = request.files['image']
    img = image.load_img(img, target_size = (299, 299))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = inception_v3.preprocess_input(x)
    x = x.copy(order="C")
    
    # encode
    x = helpers.base64_encode_image(x)
    # create a image id
    this_id = str(uuid.uuid4())
    
    d = {"id": this_id, "image": x}
    
    # push to the redis queue
    db.rpush(INCEPTIONV3_IMAGE_QUEUE, json.dumps(d))
    
    while True:
        # check if the response has been returned
        output = db.get(this_id)
        
        if output is not None:
            output = output.decode('utf-8')
            data["prediction"] = json.loads(output)
            
            db.delete(this_id)
            break
        else:
            print "* Waiting for the Inference Server..."
            time.sleep(CLIENT_SLEEP)
        
        data['success'] = True
        
    return jsonify({
        "data": data
        }), 200
        
    
    
    
    
    
    
    