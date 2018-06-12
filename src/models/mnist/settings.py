import os

MNIST_MODEL_PATH = os.path.join("models", "mnist", "mnist_baseline_less.h5")
BATCH_SIZE = 32
IMAGE_QUEUE = 'mnist_image_queue'
IMAGE_TYPE = 'float32'
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
SERVER_SLEEP = 0.5