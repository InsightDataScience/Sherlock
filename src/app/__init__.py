import os
import redis
from flask import Flask
# from apis.mnist import mnist_model_server

# TO DO Move to Settings

app = Flask(__name__)

# TO DO:
# add to config.
pool = redis.ConnectionPool(host='redis', port=6379, db=0)
db = redis.Redis(connection_pool=pool)

app.config.from_object('config')

from apis import *