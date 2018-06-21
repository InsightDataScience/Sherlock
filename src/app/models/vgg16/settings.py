import os

VGG16_MODEL_PATH = os.path.join("app", "models", "vgg16")
BATCH_SIZE = 32
IMAGE_QUEUE = 'vgg16_image_queue'
IMAGE_TYPE = 'float32'
IMAGE_SHAPE = (1, 224, 224, 3)
FC_SIZE = 1024
SERVER_SLEEP = 0.5