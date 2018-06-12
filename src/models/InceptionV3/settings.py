import os

InceptionV3_MODEL_PATH = os.path.join("models", "InceptionV3", "InceptionV3.h5")
BATCH_SIZE =32
IMAGE_QUEUE = 'inceptionV3_image_queue'
IMAGE_TYPE = 'float32'
IMAGE_SHAPE = (1, 299, 299, 3)
SERVER_SLEEP = 0.5