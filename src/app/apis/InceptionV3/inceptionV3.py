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

import celery

# keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# flask
from flask import jsonify
from flask import Blueprint, request

import API_helpers

from app import app
from app import db

# michaniki app
from ...tasks import *

blueprint = Blueprint('inceptionV3', __name__)

@blueprint.route('/retrain', methods=['POST'])
def retrain():
    """
    pick up a pre-trained model
    resume training using more data
    
    @args: train_bucket_url: URL pointing to the folder for training data on S3
    @args: model_name: the name of the model want to be retraiend, the folder must be exsit
    """
    bucket_url = request.form.get('train_bucket_url')
    model_name = request.form.get('model_name')
    local_data_path = os.path.join('/tmp/model_data/', model_name)

    # download the folder in the url
    API_helpers.download_a_dir_from_s3(bucket_url, local = local_data_path)
    
    try:
        # kick off the retraining service in celery worker
        inceptionV3_transfer_retraining.InceptionRetrainer(model_name)
        
        # TO DO:
        # Working on the re-traning 
        # Put to celery worker
        
        # delete the image folder
        shutil.rmtree(local_data_path, ignore_errors=True)
        
        return jsonify({
            "status": "success"
        }), 200
    except Exception as err:
        # delete the image folder
        shutil.rmtree(local_data_path, ignore_errors=True)
        return jsonify({
            "status": str(err)
            }), 500
    
@blueprint.route('/transfer', methods=['POST'])
def init_new_model():
    """
    init a new model based on InceptionV3
    that can predict picture for new classes.
     
    @args: train_bucket_url: URL pointing to the folder for training data on S3
    """
    # need to load the base model here
 
    s3_bucket_name = request.form.get('train_bucket_name')
    s3_bucket_prefix = request.form.get('train_bucket_prefix')
    model_name = s3_bucket_prefix.split('/')[-1]
    
    local_data_path = os.path.join('./tmp')
    
    # generate a celery task id
    this_id = celery.uuid()

    # download the folder in the url
    output_path = API_helpers.download_a_dir_from_s3(bucket_name = s3_bucket_name,
                                    bucket_prefix = s3_bucket_prefix,
                                    local_path = local_data_path)

    # kick off the transfer learning thing here
    async_transfer.apply_async((model_name, output_path, this_id), task_id=this_id)
    
    return jsonify({
        "task_id": this_id,
        "status": "Transfer Learning and Fine-Tuning are Initiated"
    }), 200

    
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
    x = preprocess_input(x)
    x = x.copy(order="C")
    
    # encode
    x = API_helpers.base64_encode_image(x)
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
#             print "* Waiting for the Inference Server..."
            time.sleep(CLIENT_SLEEP)
        
        data['success'] = True
        
    return jsonify({
        "data": data
        }), 200
        
    
    
    
    
    
    
    