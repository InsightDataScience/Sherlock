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
import botocore
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

def download_a_dir_from_s3(bucket_name, local_path):
    """
    download the folder from S3

    local: /src/tmp/model_data/

    Will not download if the local folder already exists
    """
    logging.info("* Helper: Loading Text from S3 {} ".format(bucket_name))
    path = os.path.join(bucket_name,'data')
    output_path = os.path.join(local_path, bucket_name)
    save_path = os.path.join(local_path, path)
    logging.info('*Saving text files at:%s',save_path)

    s3 = boto3.resource('s3')
    
    # if blank prefix is given, return everything)
    key1 = 'train.tsv'
    key2 = 'val.tsv'
    key3 = 'test.tsv'
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    try:
        s3.Bucket(bucket_name).download_file(key1, os.path.join(save_path,'train.tsv'))
        s3.Bucket(bucket_name).download_file(key2, os.path.join(save_path,'dev.tsv'))
        
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    logging.info("* Helper: Text Loaded at: {}".format(output_path))
    return output_path

def download_test_file_from_s3(bucket_name, local_path):
    """
    download the folder from S3

    local: /src/tmp/model_data/

    Will not download if the local folder already exists
    """
    logging.info("* Helper: Loading Text from S3 {} ".format(bucket_name))
    path = os.path.join(bucket_name,'data')
    output_path = os.path.join(local_path, bucket_name)
    save_path = os.path.join(local_path, path)
    logging.info('*Saving text files at:%s',save_path)

    s3 = boto3.resource('s3')
    mybucket = s3.Bucket(bucket_name)
    # if blank prefix is given, return everything)
    
    key3 = 'test.tsv'
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    try:
        mybucket.download_file(key3, os.path.join(save_path,'test.tsv'))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    logging.info("* Helper: Text Loaded at: {}".format(output_path))
    return output_path
