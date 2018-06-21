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
from keras_applications import vgg16

# flask
from flask import jsonify
from flask import Blueprint, request

import API_helpers

from app import app
from app import db

# michaniki app
from ...tasks import *

blueprint = Blueprint('vgg16', __name__)

@blueprint.route('/retrain', methods=['POST'])
def retrain():
    """
    pick up a pre-trained model
    resume training using more data
    
    @args: s3_bucket_name: the S3 bucket name
    @args: s3_bucket_prefix: the folder path of the data the folder must be exsit
    """
    s3_bucket_name = request.form.get('s3_bucket_name')
    s3_bucket_prefix = request.form.get('s3_bucket_prefix')
    nb_epoch = int(request.form.get('nb_epoch'))
    batch_size = int(request.form.get('batch_size'))
    model_name = s3_bucket_prefix.split('/')[-1]
    
    # TO DO:
    # check if the model is under training
    
    local_data_path = os.path.join('./tmp')

    # download the folder in the url
    output_path = API_helpers.download_a_dir_from_s3(bucket_name = s3_bucket_name,
                                bucket_prefix = s3_bucket_prefix,
                                local_path = local_data_path)
    
    # generate a task id 
    this_id = celery.uuid()
    
    # call the async task
    async_retrain.apply_async((model_name, output_path, nb_epoch, batch_size, this_id), task_id = this_id)
    
    return jsonify({
        "status": "success"
    }), 200
    
@blueprint.route('/transfer', methods=['POST'])
def init_new_model():
    """
    init a new model based on InceptionV3
    that can predict picture for new classes.
     
    @args: s3_bucket_name: the S3 bucket name
    @args: s3_bucket_prefix: the folder path of the data
    """
    s3_bucket_name = request.form.get('train_bucket_name')
    s3_bucket_prefix = request.form.get('train_bucket_prefix')
    model_name = s3_bucket_prefix.split('/')[-1]
    
    # TO DO:
    # check if the model is under training
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
    img = image.load_img(img, target_size = (224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = vgg16.preprocess_input(x)
    x = x.copy(order="C")
    
    # encode
    x = API_helpers.base64_encode_image(x)
    # create a image id
    this_id = str(uuid.uuid4())
    
    d = {"id": this_id, "image": x, "model_name": model_name}
    
    # push to the redis queue
    db.rpush(VGG16_IMAGE_QUEUE, json.dumps(d))
    
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
        
    
    
    
    
    
    
    