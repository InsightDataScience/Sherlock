'''
Created on Jun 6, 2018

define blueprints here

@author: runshengsong
'''
from flask import Flask

from .mnist import blueprint as mnist_blueprint

from app import app

app.register_blueprint(mnist_blueprint, url_prefix = '/mnist')

@app.route('/')
def index():
    return 'Welcome to michaniki'
