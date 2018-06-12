'''
Created on Jun 11, 2018

@author: runshengsong
'''
import redis
import time
import json
import numpy as np
from rq import Queue

# inception helpers
import helpers
import settings

# Keras
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils

db = redis.StrictRedis(host="localhost", port=6379,
                       db=0)

def run_inceptionV3_model_server():
    '''
    run the inference server for Inception V3
    
    Pull image from the Redis, decode 
    send to the model, predict
    return the response to the redis
    
    Images are tracked using is Image IDs
    '''
    print "* Loading InceptionV3 Model..."
    try:
        model = load_model(settings.InceptionV3_MODEL_PATH)
    except:
        model = keras.applications.inception_v3.InceptionV3(include_top=True, 
                                                            weights='imagenet',
                                                            classes=1000)
        model.save(settings.InceptionV3_MODEL_PATH)
    print "* InceptionV3 Model Loaded!"
    
    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE)
        imageIDs = []
        batch = None
        
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            
            # decode image 
            this_image = helpers.base64_decode_image(q['image'], 
                                                    settings.IMAGE_TYPE,
                                                    shape = settings.IMAGE_SHAPE)
        
            # stack up the image to the batch
            if batch is None:
                batch = this_image
            else:
                batch = np.vstack([batch, this_image])
                
            # add the image id
            imageIDs.append(q['id'])
            
        # if there is any images in the batch
        if len(imageIDs) > 0:
            print "* Batch size: {}".format(batch.shape)
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)
            
            # loop ever each image in the batch
            for (each_id, each_result) in zip(imageIDs, results):
                this_output = []
                
                # generate probability of top classes
                for (image_net_id, label, prob) in each_result:
                    r = {"label": label,
                         "probability": float(prob)}
                    
                    this_output.append(r)
                
                # add this result to the queue
                # indexed by image id
                db.set(each_id, json.dumps(this_output))
            
            # delete this image batch from the queue
            # to save space
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
        
        # sleep and wait
        time.sleep(settings.SERVER_SLEEP)
        
if __name__ == "__main__":
    run_inceptionV3_model_server()
    
            