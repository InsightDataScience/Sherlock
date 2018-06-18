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

# michaniki app
from ...models.InceptionV3 import inceptionV3_transfer_retraining

blueprint = Blueprint('inceptionV3', __name__)

INCEPTIONV3_IMAGE_QUEUE = app.config['INCEPTIONV3_IMAGE_QUEUE']
CLIENT_SLEEP = app.config['CLIENT_SLEEP']
INCEPTIONV3_TOPLESS_MODEL_PATH = app.config['INCEPTIONV3_TOPLESS_MODEL_PATH']
INV3_TRANSFER_NB_EPOCH = 3
INV3_TRANSFER_BATCH_SIZE = 2


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
    helpers.download_a_dir_from_s3(bucket_url, local = local_data_path)
    
    # TO DO:
    # Celery task
    # kick off the retraining service
    inceptionV3_transfer_retraining.InceptionRetrainer(model_name)
    
    return jsonify({
        "status": "success"
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
    
    local_data_path = os.path.join('./tmp')
    
    # download the folder in the url
    output_path = helpers.download_a_dir_from_s3(bucket_name = s3_bucket_name,
                                    bucket_prefix = s3_bucket_prefix,
                                    local_path = local_data_path)
    
    # TO DO:
    # Celery Task
    # kick off the transfer learning thing here
    new_model_path = os.path.join("app", "models", model_name)
    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)
    this_IV3_transfer = inceptionV3_transfer_retraining.InceptionTransferLeaner(model_name)
    new_model = this_IV3_transfer.transfer_model(output_path, 
                                     nb_epoch = INV3_TRANSFER_NB_EPOCH,
                                     batch_size = INV3_TRANSFER_BATCH_SIZE)
    
    new_model_path = os.path.join(new_model_path, model_name + ".h5")
    new_model.save(new_model_path)
    print "* Transfer: New Model Saved at: {}".format(new_model_path)

    return jsonify({
        "status": "success"
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
        
    
    
    
    
    
    
    