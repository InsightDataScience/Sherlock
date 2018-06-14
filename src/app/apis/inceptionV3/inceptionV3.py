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
InceptionV3_TOPLESS_MODEL_PATH = app.config['InceptionV3_TOPLESS_MODEL_PATH']
 
@blueprint.route('/init', methods=['POST'])
def init_new_model():
    """
    init a new model based on InceptionV3
    that can predict picture for new classes.
     
    @args: train_bucket_url: URL pointing to the folder for training data on S3
    """
    # need to load the base model here
    try:
        base_model = load_model(InceptionV3_TOPLESS_MODEL_PATH)
    except IOError:
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
 
    bucket_url = request.form.get('train_bucket_url')
    model_name = request.form.get('model_name')
    # download the folder in the url
    helpers.down_load_a_dir_from_s3(bucket_url, local = '/tmp/model_data/' + model_name)
    
    
    
@blueprint.route('/predict', methods=['POST'])
def run_inceptionV3():
    """
    Run the pre-trained base Inception V3 model 
    and send image to queue
    
    Listening user submitted images and 
    stack them in a Redis queue
    """
    data = {"success": False}
    
    # load model name
    model_name = request.form.get('model_name')

    # load and pre-processing image
    img = request.files['image']
    img = image.load_img(img, target_size = (299, 299))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = inception_v3.preprocess_input(x)
    x = x.copy(order="C")
    
    # encode
    x = helpers.base64_encode_image(x)
    # create a image id
    this_id = str(uuid.uuid4())
    
    d = {"id": this_id, "image": x, "model_name": model_name}
    
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
        
    
    
    
    
    
    
    