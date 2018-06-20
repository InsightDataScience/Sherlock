from __future__ import absolute_import, unicode_literals

import celery
from . import app

BROKER_URL = app.config['BROKER_URL']
BACKEND_URL=app.config['BACKEND_URL']

michaniki_celery_app = celery.Celery('app', 
                                     broker=BROKER_URL,
                                     backend=BACKEND_URL,
                                     include='app.tasks')

if __name__ == '__main__':
    michaniki_celery_app.start()
    