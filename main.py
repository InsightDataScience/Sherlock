import boto3
import requests
import os
import random
import shutil
import pickle
import time
from fileio import *
from sherlockWrapper import *
from labeling import *

def random_images_loop(model_name, file_loc, base_model='inceptionV3', N_initial=100,
                     bucket='insightai2019', ip_addr='http://127.0.0.1:3031/'):
    #main body for running random 
    output_path = './results/' + model_name
    transfer_url = ip_addr + base_model + '/transfer'
    inference_url = ip_addr + base_model + '/predict'
    status_url = ip_addr + 'tasks/info'

    class_names, file_names = loadDirectory('./' + file_loc + '/train/')
    validate_class_names, validate_file_names = loadDirectory('./' +
                                                             file_loc + '/val/')
    test_class_names, test_file_names = loadDirectory('./' + file_loc + '/test/')

    train_dict = choose_n(file_names, N_initial)

    upload_to_S3(train_dict,os.path.join('models',model_name,'train'))
    upload_to_S3(random_file_dict,os.path.join('models',model_name,'train'))    
    upload_to_S3(validate_file_names, os.path.join('models',model_name,'val'))
    
    r = train_new_model(model_name, bucket_name='insightai2019', path_prefix='models',
                         url=transfer_url)
    wait_for_training(r)
    rid = r['task_id']
    response = requests.post(status_url,json={rid:rid})
    r_acc = response.json()
    test_random = run_inference_on_dict(test_file_names, model_name)
    acc_random = []
    for k in test_random:
        acc_random.append(compute_accuracy(test_random,k))
        
    save_file_name = 'r{}.pickle'.format(0)
    pickle_results(output_path, save_file_name, [r_acc,test_random])
    return r_acc, test_random


def non_random_images_loop(model_name, file_loc, base_model='inceptionV3', N_initial=100,
                     bucket='insightai2019', ip_addr='http://127.0.0.1:3031'):
#    model_name = 'imgnet11.maxpool.584'
#    file_loc = 'imgnetmodel'
#    base_model = 'inceptionV3'
#    N_initial = 584
#    bucket = 'insightai2019'
#    ip_addr='http://127.0.0.1:3031/'
    
    output_path = './results/' + model_name
    transfer_url = ip_addr + base_model + '/transfer'
    inference_url = ip_addr + base_model + '/predict'
    status_url = ip_addr + 'tasks/info'
    retrain_url=ip_addr + 'inceptionV3/retrain'
    
#    iv3 = InceptionV3(weights='imagenet',input_shape=(299,299,3))
    iv3_topless = InceptionV3(include_top=False, weights='imagenet',pooling=max,
                              input_shape=(299,299,3))
    
    class_names, file_names = loadDirectory('./' + file_loc + '/train/')
    validate_class_names, validate_file_names = loadDirectory('./' +
                                                             file_loc + '/val/')
    class_names, test_file_names = loadDirectory('./' + file_loc + '/test/')

    file_list = []
    file_labels = []
    for k in file_names:
        file_list.extend(file_names[k])
        file_labels.extend([k] * len(file_names[k]))

    unlabeled_features = feature_extraction(file_list, iv3_topless)
    points = pick_points_faster(unlabeled_features, iv3_topless, N_initial)
    labeled_files = [file_list[idx] for idx in points]

    upload_dict = {k :[] for k in class_names}
    for idx in points:
        upload_dict[file_labels[idx]].append(file_list[idx])
    upload_to_S3(upload_dict,os.path.join('models',model_name,'train'))
    upload_to_S3(validate_file_names, os.path.join('models',model_name,'val'))
    
    r = train_new_model(model_name, bucket_name='insightai2019', path_prefix='models',
                         url=transfer_url)
    wait_for_training(r)
    rid = r['task_id']
    response = requests.post(status_url,json={rid:rid})
    train_acc = response.json()#83.6 training, 81.6 validation
    test_results = runInferenceOnDict(test_file_names, model_name)
    test_acc = []
    for k in test_results:
        test_acc.append(computeAccuracy(test_results,k))

    return train_acc, test_acc


def main(model_name, base_model='inceptionV3', N_initial=5,
         iterations=1, labelsPerRound=5, bucket='insightai2019',
         ip_addr='http://127.0.0.1:3031/'):

    model_name = 'HotWineBike1kRandom'
    base_model = 'inceptionV3'
    N_initial = 25
    iterations = 1
    labelsPerRound = 25
    bucket = 'insightai2019'
    output_path = './results/' + model_name
    ip_addr='http://127.0.0.1:3031/'
    transfer_url = ip_addr + base_model + '/transfer'
    inference_url = ip_addr + base_model + '/predict'
    status_url = ip_addr + 'tasks/info'
    retrain_url=ip_addr + 'inceptionV3/retrain'
    iv3 = InceptionV3(weights='imagenet',input_shape=(299,299,3))
    iv3_topless = InceptionV3(include_top=False, weights='imagenet',input_shape=(299,299,3))
    # load the images - array of Images
    
            
if __name__ == '__main__':
    main('tomato_potato')

