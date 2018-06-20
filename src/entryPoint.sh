#!/bin/bash
set -e

python install_base_model.py
uwsgi --ini uwsgi.ini