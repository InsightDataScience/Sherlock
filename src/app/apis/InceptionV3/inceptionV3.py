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
import glob
import shutil

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

# temp folder save image files downloaded from S3
TEMP_FOLDER = os.path.join('./tmp')

blueprint = Blueprint('inceptionV3', __name__)

@blueprint.route('/label', methods=['POST'])
def label():
    """
    given a folder in S3 bucket
    label all images in it.
    """
    s3_bucket_name = request.form.get('s3_bucket_name')
    s3_bucket_prefix = request.form.get('s3_bucket_prefix')
    model_name = request.form.get('model_name')
    
    # load image from s3
    image_data_path = API_helpers.download_a_dir_from_s3(s3_bucket_name,
                                                     s3_bucket_prefix, 
                                                     local_path = TEMP_FOLDER)
    
    # for each images in the folder
    # supports .png and .jpg
    all_image_ids = []
    all_pred = []

    for each_image in glob.glob(image_data_path + "/*.*"):
        iamge_name = each_image.split('/')[-1]
        this_img = image.load_img(each_image, target_size = (299, 299))
        
        # image pre-processing
        x = np.expand_dims(image.img_to_array(this_img), axis=0)
        x = preprocess_input(x)
        x = x.copy(order="C")
        
        # encode
        x = API_helpers.base64_encode_image(x)
        # create a image id
        this_id = str(uuid.uuid4())
        all_image_ids.append((this_id, iamge_name))
        d = {"id": this_id, "image": x, "model_name": model_name}
        
        # push to the redis queue
        db.rpush(INCEPTIONV3_IMAGE_QUEUE, json.dumps(d))
    
    all_pred = []
    while all_image_ids:
        # pop the first one from the queue
        this_id, this_image_name = all_image_ids.pop(0)
        this_pred = {}
        
        while True:
            # check if the response has been returned
            output = db.get(this_id)

            if output is not None:
                this_pred["image name"] = this_image_name
                output = output.decode('utf-8')
                this_pred["prediction"] = json.loads(output)
                
                db.delete(this_id)
                break
            else:
                time.sleep(CLIENT_SLEEP)
                
        all_pred.append(this_pred)
    
    # remove the temp folder
    shutil.rmtree(image_data_path, ignore_errors=True)
    
    return jsonify({
        "data": all_pred
        })   
        
@blueprint.route('/retrain', methods=['POST'])
def retrain():
    """
    pick up a pre-trained model
    resume training using more data
    
    @args: train_bucket_url: URL pointing to the folder for training data on S3
    @args: model_name: the name of the model want to be retraiend, the folder must be exsit
    """
    s3_bucket_name = request.form.get('train_bucket_name')
    s3_bucket_prefix = request.form.get('train_bucket_prefix')
    nb_epoch = request.form.get('nb_epoch')
    batch_size = request.form.get('batch_size')
    
    model_name = s3_bucket_prefix.split('/')[-1]
    local_data_path = os.path.join('./tmp')
    
    # create a celer task id
    this_id = celery.uuid()
    # download the folder in the url
    # return the path of the image files
    async_retrain.apply_async((model_name, 
                               local_data_path,
                               s3_bucket_name,
                               s3_bucket_prefix,
                               nb_epoch,
                               batch_size,
                               this_id), task_id=this_id)

    return jsonify({
        "task_id": this_id,
        "status": "Retraining and Fine-Tuning are Initiated"
    }), 200
    
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
    
    # generate a celery task id
    this_id = celery.uuid()

    # download the folder in the url
    # kick off the transfer learning thing here
    async_transfer.apply_async((model_name, 
                                s3_bucket_name,
                                s3_bucket_prefix,
                                this_id), task_id=this_id)
    
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
        
    
    
    
    
    
    
    