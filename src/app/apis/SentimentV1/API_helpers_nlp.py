'''
Created on Jun 8, 2018

@author: runshengsong
'''
# TO DO
# move settings to config
import os
import sys
import glob
import json
import boto3
import base64
import numpy as np
import logging

from keras.preprocessing import image


def save_classes_label_dict(label_dict, file_path_name):
    """
    save the class label dictionary to json
    """
    with open(file_path_name, 'w') as fp:
        json.dump(label_dict, fp)

    logging.info("* Helper: Classes Label Json Saved")

def download_a_dir_from_s3(bucket_name, bucket_prefix, local_path):
    """
    download the folder from S3

    local: /src/tmp/model_data/

    Will not download if the local folder already exists
    """
    logging.info("* Helper: Loading Text from S3 {} {}".format(bucket_name,bucket_prefix))
    output_path = os.path.join(local_path, bucket_prefix)
    save_path = os.path.join(local_path, path)
    if not os.path.exists(os.path.join(output_path, 'train')):
        s3 = boto3.resource('s3')
        mybucket = s3.Bucket(bucket_name)
        # if blank prefix is given, return everything)
        key1 = 'data/train.csv'
        key2 = 'data/dev.csv'
        key3 = 'data/test.csv'
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        try:
            s3.Bucket(bucket_name).download_file(key1, os.path.join(save_path,'train.csv'))
            s3.Bucket(bucket_name).download_file(key2, os.path.join(save_path,'dev.csv'))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

    logging.info("* Helper: Images Loaded at: {}".format(output_path))
    return output_path
