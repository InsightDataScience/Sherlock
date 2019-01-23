'''
Created on Jan 22, 2019

@author: manu
'''
import os
import redis
from textblob import TextBlob
import settings

pool = redis.ConnectionPool(host='redis', port=6379, db=0)
db = redis.Redis(connection_pool=pool)

class sentimentv1_inference_server:
    def __init__(self):
        # pre-load some models here on start
        self.loaded_models = {}

    def run_inceptionV3_infernece_server(self):
        '''
        run the inference server for Sentiment Analysis

        Pull sentence from the Redis, predict
        return the response to the redis

        Sentecnes are tracked using is their id
        '''
        while True:
            queue = db.lrange(settings.TEXT_QUEUE, 0, settings.BATCH_SIZE) #Is this queue different from the Queue in API path
            textIDs = defaultdict(list) #dict to hold sentence and id for a model type
            text_list=[]
            sent_list = []
            for q in queue:
                q = json.loads(q.decode("utf-8"))

                model_name = str(q['model_name'])
                id = q['id']
                sentence = q['text']

                text_list.append({"model_name":model_name,"id":id, "text":sentence})
                textIDs[model_name].append(q['id'])
                num_text += 1

                sent_list.append(sentence)


            if textIDs:
                print "* Predicting for {} of Models".format(len(textIDs.keys()))
                print "* Number of Sentences: {}".format(num_text)

                r = {"positive":0.5, "negative":0.5}
                for t in text_list:
                    preds = TextBlob(t["text"])
                    db.set(t["id"], json.dumps(preds))
