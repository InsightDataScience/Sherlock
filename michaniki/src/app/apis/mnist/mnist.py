'''
Created on Jun 6, 2018

Flask API to run minst

@author: runshengsong
'''

# move to helper
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

# flask
from flask import jsonify
from flask import Blueprint, request

import helpers

from app import app
from app import db

MNIST_IMAGE_QUEUE = app.config['MNIST_IMAGE_QUEUE']
CLIENT_SLEEP = app.config['CLIENT_SLEEP']

blueprint = Blueprint('mnist', __name__)

@blueprint.route('/predict', methods=['POST'])
def run_mnist():
    """
    Load all *.png files in a directory 
    """
    data = {"success": False}
    
    # load image
    img = request.files['image']
    img = Image.open(img)
    # pre-process
    img = helpers.pre_process_image(img)
    img = img.copy(order="C")
    
    # generate an ID for the classification then add the
    # classification ID + image to the queue
    this_id = str(uuid.uuid4())
    image = helpers.base64_encode_image(img)
    
    print len(image)
    d = {"id": this_id, "image": image}
    
    # push the current id and image to redis
    db.rpush(MNIST_IMAGE_QUEUE, json.dumps(d))
    
    while True:
        # try to get the prediction results
        output = db.get(this_id)
        
        if output is not None:
            # return it
            output = output.decode('utf-8')
            data["predictions"] = json.loads(output)
            
            # it is safe to delete the output from Redis now
            db.delete(this_id)
            break
        
        # if the output is not ready
        # just wait a bit
        print "* Waiting....."
        time.sleep(CLIENT_SLEEP)
    
    data['success'] = True
    
    return jsonify({
        "data": data
        }), 200