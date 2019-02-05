'''
Created on Jan 22, 2019

Web service for Sentiment Analysis

@author: manu
'''
import uuid
import json
import os
import io
import time
import celery
import json
import logging
# flask
from flask import jsonify
from flask import Blueprint, request

import API_helpers_nlp

from app import app
from app import db

# michaniki app
from ...tasks_nlp import async_train_bert
from ...tasks_nlp import async_test_bert

SENTIMENT_TEXT_QUEUE = app.config['SENTIMENT_TEXT_QUEUE']
CLIENT_SLEEP = app.config['CLIENT_SLEEP']
# temp folder save image files downloaded from S3
TEMP_FOLDER = os.path.join('./tmp')

blueprint = Blueprint('sentimentV1', __name__)

@blueprint.route('/predict', methods=['POST'])
def pred_sentiment():
    """
    Run the pre-trained base Sentiment analysis model
    and send sentence to queue

    Listening user submitted sentences and
    stack them in a Redis queue
    """

    logging.info("Inside pred_Sentence")
    data = {"success": False}

    model_name = 'base'

    message = request.form.get('textv')
    logging.info("Received message:%s", message)
    #sentence = Sentence(message)

    # create a image id
    this_id = str(uuid.uuid4())

    d = {"id": this_id, "text": message, "model_name": model_name}

    # push to the redis queue
    db.rpush(SENTIMENT_TEXT_QUEUE, json.dumps(d))

    while True:
        # check if the response has been returned
        output = db.get(this_id)
        if output is not None:
            output = output.decode('utf-8')
            data["prediction"] = json.loads(output)

            db.delete(this_id)
            break
        else:
            #print "* Waiting for the Sentiment Inference Server..."
            time.sleep(CLIENT_SLEEP)

        data['success'] = True
    return jsonify({
        "data": data
        }), 200

@blueprint.route('/trainbert', methods=['POST'])
def run_train_bert():
    """
    Finetune BERT uncased small language model
    """
    s3_bucket_name = request.form.get('train_bucket_name')
    #model_name = request.form.get('model_name')
    model_name = s3_bucket_name
    local_data_path = os.path.join('./tmp')
    batch_size = 32
    nb_epoch = 3

    # create a celer task id
    this_id = celery.uuid()

    async_train_bert.apply_async((model_name,
                               local_data_path,
                               s3_bucket_name,
                               nb_epoch,
                               batch_size,
                               this_id), task_id=this_id)
    return jsonify({
        "task_id": this_id,
        "status": "Retraining and Fine-Tuning usign BERT is Initiated"
    }), 200

@blueprint.route('/testbert', methods=['POST'])
def run_test_bert():
    """
    Test sentences using BERT fine tuned model
    """
    s3_bucket_name = request.form.get('test_bucket_name')
    #model_name = request.form.get('model_name')
    model_name = s3_bucket_name
    local_data_path = os.path.join('./tmp')
    batch_size = 32
    nb_epoch = 3

    # create a celer task id
    this_id = celery.uuid()

    async_test_bert.apply_async((model_name,
                               local_data_path,
                               s3_bucket_name,
                               nb_epoch,
                               batch_size,
                               this_id), task_id=this_id)
    return jsonify({
        "task_id": this_id,
        "status": "Testing usign BERT is Started. Results uploaded to S3 bucket"
    }), 200