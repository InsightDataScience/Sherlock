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
IMAGE_QUEUE = env.str('IMAGE_QUEUE', default='image_queue')
BATCH_SIZE = env.str('BATCH_SIZE', default=32)
SERVER_SLEEP = env.str('SERVER_SLEEP', default=0.5)
CLIENT_SLEEP = env.str('CLIENT_SLEEP', default=0.5)
MNIST_PATH = env.str('MNIST_PATH', default=os.path.join("app", "models", "mnist", "current", "mnist_baseline_less.h5"))