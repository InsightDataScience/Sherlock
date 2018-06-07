from flask import Flask

app = Flask(__name__)

# TO DO
# add config here
# app.config.from_object('config')

from apis import *