import os
import redis
from envparse import env

# settings_for_MNIST
MNIST_IMAGE_QUEUE = env.str('MNIST_IMAGE_QUEUE', default='mnist_image_queue')
CLIENT_SLEEP = env.str('CLIENT_SLEEP', default=0.5)

# settings for InceptionV3
INCEPTIONV3_TOPLESS_MODEL_PATH = env.str('INCEPTIONV3_TOPLESS_MODEL_PATH', default=os.path.join("app", "models", "InceptionV3", "topless",'topless.h5'))
INCEPTIONV3_IMAGE_QUEUE = env.str('INCEPTIONV3_IMAGE_QUEUE', default='inceptionV3_image_queue')
INV3_TRANSFER_NB_EPOCH = env.str('INV3_TRANSFER_NB_EPOCH', default=3)
INV3_TRANSFER_BATCH_SIZE = env.str('INV3_TRANSFER_BATCH_SIZE', default=2)

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