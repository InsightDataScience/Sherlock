import boto3
import glob
import requests
import os
import random
import shutil
import pickle
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics import pairwise_distances_argmin_min

    
def labeling_priority(data_unlabeled, trained_model, data_labeled=None):
    # order data 
    # return array of file handlers ordered by priority
    pass


def check_label_top_n(result, label, n=1):
    # returns True if label is in the top n, false otherwise
    result = result['data']['prediction'][:n]
    for r in result:
        if label in r['label']:
            return True
    return False


def choose_n_from_each_class(file_dict, n):
    ret_dict = {}
    for class_name in file_dict:
        ret_dict[class_name] = []
        for i in range(n):
            if file_dict[class_name]:
                ret_dict[class_name].append(file_dict[class_name][0])
                del file_dict[class_name][0]
    return ret_dict

            
def run_inference_on_dict(file_dict, model_name='base',
                         url='http://127.0.0.1:3031/inceptionV3/predict'):
    results = {}
    for class_name in file_dict:
        results[class_name] = []
        for dp in file_dict[class_name]:
            results[class_name].append(query_inference_server(dp, model_name,url))
    return results


def pickle_results(path,file_name,data):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(os.path.join(path,file_name),'w+')
    pickle.dump(data,f)
    f.close()
    return True



def file_dict_to_flat(file_dict):
    file_list = []
    for class_name in file_dict:
        file_list.extend( file_dict[class_name])
    return file_list


def file_list_to_dict(file_list):
    file_dict = {}
    for f in file_list:
        class_name = f.split('/')[-2]
        if class_name in file_dict:
            file_dict[class_name].append(f)
        else:
            file_dict[class_name] = [f]
    return file_dict


def magic_label(file_names, N, reserve_dict,model='base'):
    if model == 'base':
        td = choose_n_from_each_class(file_names, N-1)
        for k in reserve_dict:
            td[k].append(reserve_dict[k][0])
            del reserve_dict[k][0]
        return td


def choose_n(file_dict, n):
    ret_dict = {}
    for class_name in file_dict:
        if file_dict[class_name]:
            ret_dict[class_name] = file_dict[class_name][:n]                         
    return ret_dict


def randomly_choose_n(file_list, n):
    random.seed(90210)
    return random.sample(file_list, n)


def compute_accuracy(predictions,class_name):
    res = predictions[class_name]
    correct = sum(res[x]['data']['prediction'][0]['label'] == class_name
                  for x in range(len(res)))
    return float(correct) / len(res)

        
def feature_extraction(file_names,your_model):
#given a list of images and model, returns a list of ndarray feature weights
    feature_list = []
    for f in file_names:

        img = image.load_img(f,target_size=(299, 299))
        
        x = np.expand_dims(image.img_to_array(img), axis=0)
        x = preprocess_input(x)

        feature = your_model.predict(x)
        feature_np = np.array(feature)
        feature_list.append(feature_np.flatten())

    return feature_list


def pick_points_faster(unlabeled_features, model, n,labeled_features=[]):
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.metrics.pairwise  import euclidean_distances as ed
    labels = []
    if len(labeled_eatures) == 0:
        labeled_features = [unlabeled_features[0]]
        labels.append(0)
        n = n - 1

    indices, values = pairwise_distances_argmin_min(unlabeled_features,
                                                        labeled_features)
    for i in range(n):
        
        max_of_min = np.argmax(values)
        printme = values[max_of_min]
        labeled_features.append(unlabeled_features[max_of_min])
        labels.append(max_of_min)
        indices, values_new = pairwise_distances_argmin_min(unlabeled_features,
                                                        [labeled_features[-1]])
        for j in range(len(unlabeled_features)):
            if values_new[j] < values[j]:
                values[j] = values_new[j]

    return labels 


def cluster_label(file_names,your_model,n):
    feature_list = []
    for f in file_names:
        img = image.load_img(f,target_size=(299, 299))

        x = np.expand_dims(image.img_to_array(img), axis=0)
        x = preprocess_input(x)
#        x = x.copy(order="C")

        feature = your_model.predict(x)
        feature_np = np.array(feature)
        feature_list.append(feature_np.flatten())

    feature_list_np = np.array(feature_list)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(feature_list_np)
    return kmeans




#keras
#requests
#boto3
#tensorflow
#sklearn.metrics
