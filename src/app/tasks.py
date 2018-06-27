# coding=utf-8
from __future__ import absolute_import

from .celeryapp import michaniki_celery_app

import os
import time
import shutil

from keras.models import load_model

from app import app

from .apis.InceptionV3 import API_helpers
from .models.InceptionV3 import inceptionV3_transfer_retraining

CLIENT_SLEEP = app.config['CLIENT_SLEEP']
INV3_TRANSFER_NB_EPOCH = app.config['INV3_TRANSFER_NB_EPOCH']
INV3_TRANSFER_BATCH_SIZE = app.config['INV3_TRANSFER_BATCH_SIZE']
INCEPTIONV3_IMAGE_QUEUE = app.config['INCEPTIONV3_IMAGE_QUEUE']
INCEPTIONV3_TOPLESS_MODEL_PATH = app.config['INCEPTIONV3_TOPLESS_MODEL_PATH']
TEMP_FOLDER = os.path.join('./tmp')

@michaniki_celery_app.task()
def async_retrain(model_name,
                  s3_bucket_name,
                  s3_bucket_prefix,
                  nb_epoch,
                  batch_size,
                  id):
    """
    retrain model
    resume training
    """    
    # download image data to local 
    image_data_path = API_helpers.download_a_dir_from_s3(s3_bucket_name,
                                                     s3_bucket_prefix, 
                                                     local_path = TEMP_FOLDER)
    try:
        this_model_path = os.path.join("app", "models", "InceptionV3", model_name, model_name + ".h5")
        # load the model
        this_model = load_model(this_model_path)
        
        this_retrainer = inceptionV3_transfer_retraining.InceptionRetrainer(model_name)
        
        # return the retraiend new model
        new_model, history = this_retrainer.retrain(this_model,
                                           image_data_path, 
                                           nb_epoch, 
                                           batch_size)
        
        print "* Celery Transfer: Retrained Model Saved at: {}".format(this_model_path)
        # replace the current model
        new_model.save(this_model_path)
        
        # remove the local image path
        shutil.rmtree(image_data_path, ignore_errors=True)
        
        final_trn_acc = history.history['acc'][-1]
        final_val_acc = history.history['val_acc'][-1]
        return final_trn_acc, final_val_acc
    except Exception as err:
        # remove the local image path
        shutil.rmtree(image_data_path, ignore_errors=True)
        raise
    
@michaniki_celery_app.task()
def async_transfer(model_name,
                s3_bucket_name,
                s3_bucket_prefix,
                id):
    """
    do transfer learning
    """
    # create a subfolder for this model if not exist
    new_model_folder_path = os.path.join("app", "models", "InceptionV3", model_name)
    if not os.path.exists(new_model_folder_path):
        os.makedirs(new_model_folder_path)
    
    image_data_path = API_helpers.download_a_dir_from_s3(bucket_name = s3_bucket_name,
                                    bucket_prefix = s3_bucket_prefix,
                                    local_path = TEMP_FOLDER)
    try:
        # init the transfer learning manager
        this_IV3_transfer = inceptionV3_transfer_retraining.InceptionTransferLeaner(model_name)
        new_model, label_dict, history = this_IV3_transfer.transfer_model(image_data_path, 
                                         nb_epoch = INV3_TRANSFER_NB_EPOCH,
                                         batch_size = INV3_TRANSFER_BATCH_SIZE)
        
        # save the model .h5 file and the class label file
        new_model_path = os.path.join(new_model_folder_path, model_name + ".h5")
        new_label_path = os.path.join(new_model_folder_path, model_name + ".json")
        new_model.save(new_model_path)
        API_helpers.save_classes_label_dict(label_dict, new_label_path)
        print "* Celery Transfer: New Model Saved at: {}".format(new_model_path)
        
        # delete the image folder here:
        shutil.rmtree(image_data_path, ignore_errors=True)
        
        # return the train and val acc:
        final_trn_acc = history.history['acc'][-1]
        final_val_acc = history.history['val_acc'][-1]
        return final_trn_acc, final_val_acc
    
    except Exception as err:
        # catch any error
        shutil.rmtree(new_model_folder_path, ignore_errors=True)
        shutil.rmtree(image_data_path, ignore_errors=True)
        raise
    
        