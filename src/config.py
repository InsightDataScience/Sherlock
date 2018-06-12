import os
import redis
from envparse import env

# REDIS DB
# to be updated when running in docker
REDIS_DB = env.str('REDIS_DB', default = redis.StrictRedis(host="localhost",
    port=6379, db=0))

REDIS_HOST = env.str('REDIS_HOST', default = "localhost")
REDIS_PORT = env.str('REDIS_PORT', default = 6379)
REDIS_DB = env.str('REDIS_DB', default = 0)

# settings_for_MNIST
MNIST_IMAGE_QUEUE = env.str('MNIST_IMAGE_QUEUE', default='mnist_image_queue')
CLIENT_SLEEP = env.str('CLIENT_SLEEP', default=0.5)

# settings for InceptionV3
INCEPTIONV3_IMAGE_QUEUE = env.str('INCEPTIONV3_IMAGE_QUEUE', default='inceptionV3_image_queue')
