# Michaniki

TO DO:
* Pull image from Redis DB and send to predictor.
* Wrap everything in Docker + Docker compose

---
## Development
Core packages:
* Python 2.7.3 (anaconda recommended)
* uWSGI
* Flask

Tensorflow + Keras version to be added

1. Create a virtual environment use Conda:
``` conda create --name [YOUR_ENV_NAME] python=2.7 ```
``` source activate [YOUR_ENV_NAME]```

2. Install necessary packages:
```pip install -r requirements.txt```

After done these run: ```python run_model.py``` under the `/src` folder. If you see the output `3`, meaning that your tensorflow and keras packages are correct.

## Start Web Server

Michaniki uses Flask + uWSGI and Python 2.7 as the scaffold for the web service.

To start, run:
```uwsgi --ini uwsgi.ini```
and go to `127.0.0.1/3031` you should see the welcome message
and go to `127.0.0.1/3031/mnist/predict` you should see the outputs for the `MNIST` model. (NOT FINISHED YET)


