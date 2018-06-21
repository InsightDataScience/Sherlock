'''
Created on Jun 6, 2018

define blueprints here

@author: runshengsong
'''
from flask import Flask

from .vgg16 import blueprint as vgg16_blueprint

from app import app

app.register_blueprint(vgg16_blueprint, url_prefix = '/vgg16')

@app.route('/')
def index():
    return 'Welcome to Michaniki'

