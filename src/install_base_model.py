import os
import redis
import logging
import requests
import zipfile, io
from tqdm import tqdm
import math
import boto3
import botocore
from keras.applications.inception_v3 import InceptionV3

BASE_MODEL_PATH = os.path.join("app", "models", "InceptionV3", "base", "base.h5")
TOPLESS_MODEL_PATH = os.path.join("app", "models", "InceptionV3", "topless")
BERT_MODEL_PATH = os.path.join("app", "models", "SentimentV1", "uncased_L-12_H-768_A-12","bert_model.ckpt.data-00000-of-00001")
BERT_DIR_PATH = os.path.join("app", "models", "SentimentV1")
FAST_IMDB_DIR = os.path.join("app", "models", "SentimentV1","fastimdb")
FAST_IMDB_MODEL_PATH = os.path.join("app", "models", "SentimentV1","fastimdb","imdb_model.bin")

# loading base model
if os.path.exists(BASE_MODEL_PATH):
    print "* Starting: Found Base Model."
else:
    print "* Starting: No Base Model Found. Loading..."
    base_model = InceptionV3(include_top=True, weights='imagenet',input_shape=(299, 299, 3))

    base_model.save(BASE_MODEL_PATH)
    print "* Starting: Base Model Saved!"

# loading topless model
if os.path.exists(TOPLESS_MODEL_PATH):
    print "* Starting: Found Topless Model."
else:
    os.makedirs(TOPLESS_MODEL_PATH)
    print "* Starting: No Topless Model Found. Loading..."
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    base_model.save(os.path.join(TOPLESS_MODEL_PATH, "topless.h5"))
    print "* Starting: Topless Model Saved!"

if os.path.exists(BERT_MODEL_PATH):
    logging.info("* Found BERT uncased model")
else:
    logging.info("BERT model not found. Downloading....")
    BERT_UNCASED_URL = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
    r = requests.get(BERT_UNCASED_URL, stream=True)
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    wrote=0
    with open(os.path.join(BERT_DIR_PATH,'uncased_L-12_H-768_A-12.zip'), 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)

    z = zipfile.ZipFile(os.path.join(BERT_DIR_PATH,'uncased_L-12_H-768_A-12.zip'))
    z.extractall(BERT_DIR_PATH)
    os.remove(os.path.join(BERT_DIR_PATH,'uncased_L-12_H-768_A-12.zip'))

if os.path.exists(FAST_IMDB_MODEL_PATH):
    logging.info("* Found Base IMDB model")
else:
    logging.info("Base IMDB model not found. Downloading....")
    s3 = boto3.resource('s3')
    mybucket = s3.Bucket('fastimdb')
    model_name = 'imdb_model.bin'
    try:
        os.makedirs(FAST_IMDB_DIR)
    except OSError:
        pass
    try:
        mybucket.download_file(model_name, os.path.join(FAST_IMDB_DIR,'imdb_model.bin'))
    except Exception as err:
        logging.info(err)

# clean up the died images upon start:
# need to wait a bit for redis container to start up
pool = redis.ConnectionPool(host='redis', port=6379, db=0)
db = redis.Redis(connection_pool=pool)
db.flushall()
