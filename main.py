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
    model_name = 'imgnet11.584'
    model_name = 'imgnet11.maxpool.584'
    file_loc = 'imgnetmodel'
    base_model = 'inceptionV3'
    N_initial = 584
    bucket = 'insightai2019'
    ip_addr='http://127.0.0.1:3031/'
    
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
    rid = r_random_1k['task_id']
    response = requests.post(status_url,json={rid:rid})
    random1k_train_acc = response.json()#83.6 training, 81.6 validation
    acc584_max = []
    for k in test_584_max:
        acc584_max.append(computeAccuracy(test_584_max,k))



        
    #74,72,100 test accuracy 1k non random
    wait_for_training(r)
    rid = train1k['task_id']
    response = requests.post(status_url,json={rid:rid})
    train1kAcc = response.json()#83.6 training, 81.6 validation
    test1kRes = runInferenceOnDict(test_file_names, model_name)

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
    
    class_names, file_names = loadDirectory('./' + model_name + '/train/')
    validate_class_names, validate_file_names = loadDirectory('./' +
                                                             model_name + '/val/')
    class_names, test_file_names = loadDirectory('./' + model_name + '/test/')
    N_initial = 333
    reserve_dict = {x : file_names[x][:iterations] for x in class_names}
    file_names   = {x : file_names[x][iterations:] for x in class_names}

    train_dict = magicLabel(file_names, N_initial, reserve_dict,'base')
    labeled_dict = {0: train_dict}

    uploadToS3(train_dict,os.path.join('models',model_name,'train'))
    uploadToS3(validate_file_names, os.path.join('models',model_name,'val'))

    file_list = []
    file_labels = []
    for k in file_names:
        file_list.extend(file_names[k])
        file_labels.extend([k] * len(file_names[k]))
    unlabeled_features = feature_extraction(file_list, iv3_topless)
    r1000 = pickPoints(unlabeled_features, iv3_topless, 1000)
    labeled_files = [file_list[idx] for idx in r1]
    labeled_features = feature_extraction(labeled_files, iv3_topless)

    r2 = pick_points(unlabeled_features, iv3_topless, 20, labeled_features=labeled_features)
    labeledFiles.extend([file_list[idx] for idx in r2])
    labeledFeatures = feature_extraction(labeledFiles, iv3_topless)

    r3 = pickPoints(unlabeledFeatures, iv3_topless, 200, labeled_features=labeled_features)
    labeled_files.extend([file_list[idx] for idx in r3])
    labeled_features = feature_extraction(labeled_files, iv3_topless)

    upload_dict = {k :[] for k in class_names}
    for idx in r1:
        upload_dict[file_labels[idx]].append(file_list[idx])
    uploadToS3(uploadDict,os.path.join('models',model_name,'train'))
    uploadToS3(validate_file_names, os.path.join('models',model_name,'val'))
    
    r = trainNewModel(model_name, bucket_name='insightai2019', path_prefix='models',
                         url=transfer_url)
    rRandom1k = trainNewModel(model_name, bucket_name='insightai2019', path_prefix='models',
                         url=transfer_url)
    rid = rRandom1k['task_id']
    response = requests.post(status_url,json={rid:rid})
    random1kTrainAcc = response.json()#83.6 training, 81.6 validation
    test1kRandom = runInferenceOnDict(test_file_names, model_name)
    acc1kRandom = []
    for k in test1kRandom:
        acc1kRandom.append(computeAccuracy(test1kRandom,k))

    #74,72,100 test accuracy 1k non random
    wait_for_training(r)
    rid = train1k['task_id']
    response = requests.post(status_url,json={rid:rid})
    train1kAcc = response.json()#83.6 training, 81.6 validation
    test1kRes = runInferenceOnDict(test_file_names, model_name)

    random_model_name = 'HotWineBike'
    test2kRandom = runInferenceOnDict(test_file_names, random_model_name)
    accRandom = []
    for k in test2kRandom:
        accRandom.append(computeAccuracy(test2kRandom,k))
    res = {0: runInferenceOnDict(test_file_names, model_name)}

    saveFileName = 'r{}.pickle'.format(0)
    pickleResults(output_path, saveFileName, [res,labeled_dict])
    backup = {x : file_names[x][:] for x in file_names}

    for i in range(iterations-1):
        rt_path = 'rt/' + model_name + '-' + str(i+1)+'/models'
        train_dict = magicLabel(file_names, N_initial, reserve_dict,'base')
        labeled_dict[i+1] = train_dict
        uploadToS3(train_dict,os.path.join(rt_path,model_name,'train'))
        uploadToS3(validate_file_names,os.path.join(rt_path,model_name,'val'))
        print rt_path
        r = retrainModel(model_name,rt_path)
        wait_for_training(r)
        res[i+1] = runInferenceOnDict(test_file_names,model_name)

        saveFileName = 'r{}.pickle'.format(i+1)
        pickleResults(output_path, saveFileName, [res,labeled_dict])
            
if __name__ == '__main__':
    main('tomato_potato')

