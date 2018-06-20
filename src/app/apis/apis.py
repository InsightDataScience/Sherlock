'''
Created on Jun 6, 2018

define blueprints here

@author: runshengsong
'''
from flask import Flask

from .mnist import blueprint as mnist_blueprint
from .inceptionV3 import blueprint as incept_blueprint

from app import app

app.register_blueprint(mnist_blueprint, url_prefix = '/mnist')
app.register_blueprint(incept_blueprint, url_prefix = '/inceptionV3')

@app.route('/')
def index():
    return 'Welcome to Michaniki'

@app.route('/add')
def add_a():
    res = add.delay(3, 4)
    
    
