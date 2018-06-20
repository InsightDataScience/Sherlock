from __future__ import absolute_import, unicode_literals

import celery
from . import app

BROKER_URL = app.config['BROKER_URL']
BROKER_URL = 'amqp://admin:mypass@rabbit_broker:5672'
BACKEND='rpc://'

michaniki_celery_app = celery.Celery('app', 
                                     broker=BROKER_URL,
                                     backend=BACKEND)

if __name__ == '__main__':
    michaniki_celery_app.start()
    