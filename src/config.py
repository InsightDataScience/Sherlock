import os
import redis
from envparse import env

CLIENT_SLEEP = env.str('CLIENT_SLEEP', default=0.5)

PATH_TO_SAVE_MODELS = env.str('PATH_TO_SAVE_MODELS', default=os.path.join("app", "models", "vgg16"))

# settings for InceptionV3
VGG16_TOPLESS_MODEL_PATH = env.str('VGG16_TOPLESS_MODEL_PATH', default=os.path.join("app", "models", "vgg16", "topless",'topless.h5'))
VGG16_IMAGE_QUEUE = env.str('VGG16_IMAGE_QUEUE', default='vgg16_image_queue')
VGG16_TRANSFER_NB_EPOCH = env.str('VGG16_TRANSFER_NB_EPOCH', default=3)
VGG16_TRANSFER_BATCH_SIZE = env.str('VGG16_TRANSFER_BATCH_SIZE', default=2)

# setting for mysql db
# parsed from environment variables
DB_HOST = env.str('DB_HOST', default='127.0.0.1')
DB_PORT = env.int('DB_PORT', default=3306)
DB_USERNAME = env.str('DB_USERNAME', default='root')
DB_PASSWORD = env.str('DB_PASSWORD', default='michaniki')
DB_NAME = env.str('DB_NAME', default='michanikidb')

# redis url for celery
BROKER_URL = env.str('BROKER_URL', default='redis://redis:6379/0')
BACKEND_URL = env.str('BACKEND_URL', default='redis://redis:6379/0')