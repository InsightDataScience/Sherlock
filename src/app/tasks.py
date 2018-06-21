# coding=utf-8
from __future__ import absolute_import

from .celeryapp import michaniki_celery_app

import os
import time
import shutil

from app import app

from .apis.vgg16 import API_helpers
from .models.vgg16 import vgg16_transfer_retraining

CLIENT_SLEEP = app.config['CLIENT_SLEEP']
VGG16_TRANSFER_NB_EPOCH = app.config['VGG16_TRANSFER_NB_EPOCH']
VGG16_TRANSFER_BATCH_SIZE = app.config['VGG16_TRANSFER_BATCH_SIZE']
VGG16_IMAGE_QUEUE = app.config['VGG16_IMAGE_QUEUE']
VGG16_TOPLESS_MODEL_PATH = app.config['VGG16_TOPLESS_MODEL_PATH']

@michaniki_celery_app.task()
def async_retrain(model_name, output_path, nb_epoch, batch_size, id):
    """
    resume training an existing model
    """
    try:
        # kick off the retraining service in celery worker
        this_retrainer = vgg16_transfer_retraining.Vgg16Retrainer(model_name)
        
        new_model, model_path = this_retrainer.retrain(output_path, nb_epoch, batch_size)
        
        # save the new model
        new_model.save(model_path)
        
        # delete the image folder
        shutil.rmtree(output_path, ignore_errors=True)
    except Exception as err:
        # delete the image folder
        shutil.rmtree(output_path, ignore_errors=True)
        raise
        
@michaniki_celery_app.task()
def async_transfer(model_name, output_path, id):
    """
    do transfer learning
    """
    # create a subfolder for this model if not exist
    new_model_folder_path = os.path.join("app", "models", "vgg16", model_name)
    if not os.path.exists(new_model_folder_path):
        os.makedirs(new_model_folder_path)
    try:
        # init the transfer learning manager
        this_transfer = vgg16_transfer_retraining.Vgg16TransferLeaner(model_name)
        new_model, label_dict = this_transfer.transfer_model(output_path, 
                                         nb_epoch = VGG16_TRANSFER_NB_EPOCH,
                                         batch_size = VGG16_TRANSFER_BATCH_SIZE)
        
        # save the model .h5 file and the class label file
        new_model_path = os.path.join(new_model_folder_path, model_name + ".h5")
        new_label_path = os.path.join(new_model_folder_path, model_name + ".json")
        
        new_model.save(new_model_path)
        API_helpers.save_classes_label_dict(label_dict, new_label_path)
        print "* Celery Transfer: New Model Saved at: {}".format(new_model_path)
        
        # delete the image folder here:
#         shutil.rmtree(output_path, ignore_errors=True)
    except Exception as err:
        # catch any error
        shutil.rmtree(new_model_folder_path, ignore_errors=True)
#         shutil.rmtree(output_path, ignore_errors=True)
        raise
        