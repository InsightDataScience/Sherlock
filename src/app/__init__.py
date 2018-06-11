import redis
from flask import Flask
# from apis.mnist import mnist_model_server

# TO DO Move to Settings

app = Flask(__name__)

db = redis.StrictRedis(host="localhost", port=6379,
                       db=0)

app.config.from_object('config')

from apis import *