'''
Created on Jan 22, 2019

@author: manu
'''
import os
import redis
import time
import json
from textblob import TextBlob
import logging
from collections import defaultdict
from flair.models import TextClassifier
from flair.data import Sentence

#helpers
import settings

pool = redis.ConnectionPool(host='redis', port=6379, db=0)
db = redis.Redis(connection_pool=pool)

class sentimentV1_inference_server:
    def __init__(self):
        # pre-load some models here on start
        self.loaded_models = {}

    def run_sentimentV1_infernece_server(self):
        '''
        run the inference server for Sentiment Analysis

        Pull sentence from the Redis, predict
        return the response to the redis

        Sentecnes are tracked using is their id
        '''
        logging.info("Sentiment Inference Server running")
        while True:
            queue = db.lrange(settings.TEXT_QUEUE, 0, settings.BATCH_SIZE) #Is this queue different from the Queue in API path
            textIDs = defaultdict(list) #dict to hold sentence and id for a model type
            text_list=[]
            sent_list = []
            num_text=0
            for q in queue:
                q = json.loads(q.decode("utf-8"))

                model_name = str(q['model_name'])
                id = q['id']
                sentence = q['text']
                logging.info("Sentence in server:%s", sentence)

                text_list.append({"model_name":model_name,"id":id, "text":sentence})
                textIDs[model_name].append(q['id'])
                num_text += 1

                sent_list.append(sentence)


            if textIDs:
                logging.info("* Predicting for {} of Models".format(len(textIDs.keys())))
                logging.info("* Number of Sentences: {}".format(num_text))

                
                for t in text_list:
                    logging.info("Text is:%s",t["text"])
                    preds = TextBlob(t["text"])
                    res = {"polarity":preds.sentiment.polarity,"subjectvity":preds.sentiment.subjectivity}
                    db.set(t["id"], json.dumps(res))

                db.ltrim(settings.TEXT_QUEUE, len(textIDs), -1)

            # sleep and wait
            time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    this_server = sentimentV1_inference_server()
    this_server.run_sentimentV1_infernece_server()
