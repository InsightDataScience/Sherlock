import os

InceptionV3_MODEL_PATH = os.path.join("app", "models", "InceptionV3")
BATCH_SIZE = 32
IMAGE_QUEUE = 'inceptionV3_image_queue'
IMAGE_TYPE = 'float32'
IMAGE_SHAPE = (1, 299, 299, 3)
FC_SIZE = 1024
SERVER_SLEEP = 0.5
NUM_LABEL_TO_RETURN = 5