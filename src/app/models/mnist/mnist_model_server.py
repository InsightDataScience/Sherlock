'''
Created on Jun 10, 2018

@author: runshengsong
'''
import redis
from rq import Queue
import time
import json
import numpy as np

# Michaniki MNIST helper
import helpers
import settings

# Keras
from keras.models import load_model

db = redis.StrictRedis(host="localhost", port=6379,
                       db=0)

def run_mnist_model_server():
    """
    The model server
    
    Pull image from the Redis, decode 
    send to the model, predict
    return the response to the redis
    
    Images are tracked using is Image IDs
    """
    print "* Loading MNIST Model..."
    mnist_model = load_model(settings.MNIST_MODEL_PATH)
    print "* MNIST Model Loaded!"
    
    while True:
        # continues listening...
        # grab a list of image (equal to the batch size from the 
        # redis db
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE)
        imageIDs = []
        batch = None
        
        # for each image in the queue
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            # decode the image
            this_image = helpers.base64_decode_image(q['image'], 
                                                    settings.IMAGE_TYPE, 
                                                    shape=(1, 1, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            # stack up the image to the current 
            # batch
            if batch is None:
                batch = this_image
            else:
                batch = np.vstack([batch, this_image]) 
            
            # add the id
            imageIDs.append(q['id'])
        
        # start to put the result back to the db
        if len(imageIDs) > 0:
            print "* Batch size: {}".format(batch.shape)
            
            # get the prediction results
            this_preds = mnist_model.predict(batch)
            
            for (each_id, each_pred) in zip(imageIDs, this_preds):
                output = [{"label": np.argmax(each_pred)}]
                # push the results to db
                # imageID as the key
                db.set(each_id, json.dumps(output))
                
            # remove the set of images from the queue
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
        
        # wait for the redis to receive new images to predict
        time.sleep(settings.SERVER_SLEEP)

if __name__ == '__main__':
    run_mnist_model_server()
    