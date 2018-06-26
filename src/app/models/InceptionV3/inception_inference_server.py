'''
Created on Jun 11, 2018

@author: runshengsong
'''
import os
import redis
import time
import json
import numpy as np
from functools import partial
from rq import Queue
from collections import defaultdict

# inception helpers
import INV3_helpers
import settings

# Keras
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils

# db = redis.StrictRedis(host="redis://redis", port=6379,
#                        db=0)
pool = redis.ConnectionPool(host='redis', port=6379, db=0)
db = redis.Redis(connection_pool=pool)

class inceptionV3_infernece_server:
    def __init__(self):
        # pre-load some models here on start
        self.loaded_models = {}
    
    def run_inceptionV3_infernece_server(self):
        '''
        run the inference server for Inception V3
        
        Pull image from the Redis, decode 
        send to the model, predict
        return the response to the redis
        
        Images are tracked using is Image IDs
        '''
        while True:
            queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE)
            imageIDs = defaultdict(list)
            batch_for_each_model = defaultdict(partial(np.ndarray, 0))
            num_pic = 0
            
            for q in queue:
                q = json.loads(q.decode("utf-8"))
                
                # decode image 
                this_image = INV3_helpers.base64_decode_image(q['image'], 
                                                        settings.IMAGE_TYPE,
                                                        shape = settings.IMAGE_SHAPE)
                
                model_to_go = str(q['model_name'])
                # stack up the image to the batch
                # for each model
                if batch_for_each_model[model_to_go].size == 0:
                    # if it is a empty array
                    batch_for_each_model[model_to_go] = this_image
                else:
                    # vstack on it
                    batch_for_each_model[model_to_go] = np.vstack([batch_for_each_model[model_to_go],
                                                                  this_image])
                    
                # add the image id
                imageIDs[model_to_go].append(q['id'])
                num_pic += 1
                
            # if there is any images in the batch
            if imageIDs:
                print "* Predicting for {} of Models".format(len(imageIDs.keys()))
                print "* Number of Picture: {}".format(num_pic)
                
                # loop over each model and predict their batch
                for each_model_name, each_batch in batch_for_each_model.iteritems():
                    this_ids = imageIDs[each_model_name] # these are the ids for the batch for this model
                    
                    # load model here
                    # check the model if already exsit
                    if each_model_name in self.loaded_models.keys():
                        model = self.loaded_models[each_model_name]
                        print "* Loaded {} Model from Mem....".format(each_model_name)
                    else:
                        # load a fresh new model
                        print "* Loading {} Model...".format(each_model_name)
                        model = load_model(os.path.join(settings.InceptionV3_MODEL_PATH, each_model_name, each_model_name+'.h5'))
                        self.loaded_models[each_model_name] = model# save the model instance
                        print "* {} Loaded and Saved in Mem.".format(each_model_name)
                    
                    # start predicting
                    preds = model.predict(each_batch)
    
                    # TO DO:
                    # Decode prediction to get the class label
                    results = INV3_helpers.decode_pred_to_label(preds, each_model_name, num_return = settings.NUM_LABEL_TO_RETURN)
                    
                    # loop ever each image in the batch
                    for (each_id, each_result) in zip(this_ids, results):
                        this_output = []
                        
                        # generate probability of top classes
                        for (label, prob) in each_result:
                            r = {"label": label,
                                 "probability": float(prob)}
                            
                            this_output.append(r)
                        
                        # add this result to the queue
                        # indexed by image id
                        db.set(each_id, json.dumps(this_output))
                        print "* Prediction for {} Sent Back!".format(each_model_name)
                
                # delete this image batch from the queue
                # to save space
                db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
            
            # sleep and wait
            time.sleep(settings.SERVER_SLEEP)
        
if __name__ == "__main__":
    this_server = inceptionV3_infernece_server()
    this_server.run_inceptionV3_infernece_server()
    
            