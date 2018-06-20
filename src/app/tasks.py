# coding=utf-8
from __future__ import absolute_import

from .celeryapp import michaniki_celery_app

import time

@michaniki_celery_app.task()
def add(x, y):
    return x + y
    