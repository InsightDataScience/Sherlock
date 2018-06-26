'''
Created on Jun 6, 2018

define blueprints here

@author: runshengsong
'''
from flask import Flask

from .mnist import blueprint as mnist_blueprint
from .InceptionV3 import blueprint as incept_blueprint
from .tasks import blueprint as tasks_blueprint

from app import app

app.register_blueprint(mnist_blueprint, url_prefix = '/mnist')
app.register_blueprint(incept_blueprint, url_prefix = '/inceptionV3')
app.register_blueprint(tasks_blueprint, url_prefix='/tasks')

@app.route('/')
def index():
    return 'Welcome to Michaniki'

@app.route('/add')
def add_a():
    res = add.delay(3, 4)
    
    
