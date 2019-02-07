import boto3
import glob
import requests
import os
import random
import shutil
import pickle
import time
    
def labelingPriority(data_unlabeled, trained_model, data_labeled=None):
    # order data 
    # return array of file handlers ordered by priority
    pass


def checkLabelTopN(result, label, n=1):
    # returns True if label is in the top n, false otherwise
    result = result['data']['prediction'][:n]
    for r in result:
        if label in r['label']:
            return True
    return False


def trainTestValidate(path, model_path, nTest=50, nVal=50,
                      names=['test', 'val', 'train'] ):
    class_name = os.path.split(path)[-1]
    file_names = glob.glob(os.path.join(path,'*'))
    destinations = map(lambda x: os.path.join(model_path, x,
                                              class_name), names)
    random.shuffle(file_names)
    for d in destinations:
        if not os.path.isdir(d):
            os.makedirs(d)
        
    map(lambda x: shutil.move(x, os.path.join(destinations[0],
                            os.path.split(x)[-1])), file_names[0:nTest] )
    map(lambda x: shutil.move(x, os.path.join(destinations[1],
                        os.path.split(x)[-1])), file_names[nTest:nTest + nVal] )
    map(lambda x: shutil.move(x, os.path.join(destinations[2],
                            os.path.split(x)[-1])), file_names[nTest + nVal:] )
    

    
def loadDirectory(path):
    # all directory names in path are class names
    # all files inside a directory share label
    class_paths = glob.glob(path + '/*')
    class_names = list(map(lambda x: os.path.split(x)[-1], class_paths))
    #there might be a bug in the following line...maybe path.join is better
    file_names = {x: glob.glob(os.path.join(path,x,'*') for x in class_names}
    return class_names, file_names


def queryInferenceServer(fileName, model_name='base',
                         url='http://127.0.0.1:3031/inceptionV3/predict'):
    form_data = {'model_name': model_name}
    files = {'image': open(fileName, 'rb')}
    response = requests.post(url, files=files, data=form_data)
    return response.json()


def trainNewModel(model_name, bucket_name='insightai2019', path_prefix='models',
                         url='http://127.0.0.1:3031/inceptionV3/transfer'):
    form_data = {
        'train_bucket_name': bucket_name, 
        'train_bucket_prefix': os.path.join(path_prefix, model_name)
    }

    response = requests.post(url, data=form_data)
    return response.json()



def checkStatus(id, url='http://127.0.0.1:3031/tasks/info'):
    response = requests.post(url,json={id:id})
    return response.json()['Tasks Status'][0]['status'] == 'SUCCESS'


def retrainModel(model_name, path, bucket_name='insightai2019',
                 nb_epoch=3, batch_size=2,
                 url='http://127.0.0.1:3031/inceptionV3/retrain'):
    form_data = {
        'nb_epoch': nb_epoch,
        'batch_size': batch_size,
        'train_bucket_name': bucket_name,
        'train_bucket_prefix': os.path.join(path, model_name)
    }

    response = requests.post(url, data=form_data)
    return response.json()

    
def uploadToS3(file_dict, key_path, bucket_name='insightai2019'):
    # push all files in file_dict to S3
    s3 = boto3.client('s3')
    for key in file_dict:
        for datapoint in file_dict[key]:
            fname= os.path.split(datapoint)[-1]
            file_key = os.path.join(key_path,key,fname)
            print file_key
            s3.upload_file(datapoint, bucket_name, file_key)


def chooseNFromEachClass(file_dict, n):
    ret_dict = {}
    for class_name in file_dict:
        ret_dict[class_name] = []
        for i in range(n):
            if file_dict[class_name]:
                ret_dict[class_name].append(file_dict[class_name][0])
                del file_dict[class_name][0]
    return ret_dict

            
def runInferenceOnDict(file_dict, model_name='base',
                         url='http://127.0.0.1:3031/inceptionV3/predict'):
    results = {}
    for class_name in file_dict:
        results[class_name] = []
        for dp in file_dict[class_name]:
            results[class_name].append(queryInferenceServer(dp, model_name,url))
    return results


def pickleResults(path,fileName,data):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(os.path.join(path,fileName),'w+')
    pickle.dump(data,f)
    f.close()
    return True

def wait_for_training(response, t=20, t_max=900,
                      url='http://127.0.0.1:3031/tasks/info'):
    status = checkStatus(response['task_id'],url)
    while not status:
        time.sleep(t)
        t += t / 10
        status = checkStatus(response['task_id'],url)
    return 1


def magicLabel(file_names, N, reserve_dict,model='base'):
    if model == 'base':
        td = chooseNFromEachClass(file_names, N-1)
        for k in reserve_dict:
            td[k].append(reserve_dict[k][0])
            del reserve_dict[k][0]
        return td

    
def main(model_name, base_model='inceptionV3', N_initial=5,
         iterations=3, labelsPerRound=5, bucket='insightai2019',
         ip_addr='http:127.0.0.1:3031'):
#    model_name = 'tomato_potato'
    model_name = 'imgnetmodel'
    base_model = 'inceptionV3'
    N_initial = 25
    iterations = 8
    labelsPerRound = 25
    bucket = 'insightai2019'
    output_path = './results/' + model_name
    ip_addr='http://127.0.0.1:3031/'
    transfer_url = ip_addr + base_model + '/transfer'
    inference_url = ip_addr + base_model + '/predict'
    status_url = ip_addr + 'tasks/info'
    retrain_url=ip_addr + 'inceptionV3/retrain'
    queryInferenceServer('hotdog.jpg', model_name='base',model_inference_url)
    # load the images - array of Images
                  
    class_names, file_names = loadDirectory('./' + model_name + '/train/')
    validate_class_names, validate_file_names = loadDirectory('./' +
                                                             model_name + '/val/')
    class_names, test_file_names = loadDirectory('./' + model_name + '/test/')

    reserve_dict = {x : file_names[x][:iterations] for x in class_names}
    file_names   = {x : file_names[x][iterations:] for x in class_names}
    
    # first training pass takes N_initial of each class

    train_dict = magicLabel(file_names, N_initial, reserve_dict,'base')
    labeled_dict = {0: train_dict}

    uploadToS3(train_dict,os.path.join('models',model_name,'train'))
    uploadToS3(validate_file_names, os.path.join('models',model_name,'val'))

    

    r = trainNewModel(model_name)
    wait_for_training(r)
    res = {0: runInferenceOnDict(test_file_names, model_name)}

    saveFileName = 'r{}.pickle'.format(0)
    pickleResults(output_path, saveFileName, [res,labeled_dict])
    

    for i in range(iterations):
        rt_path = 'rt/' + model_name + '-' + str(i+1)+'/models'
        train_dict = magicLabel(file_names, N_initial, reserve_dict,'base')
        labeled_dict[i+1] = train_dict
        uploadToS3(train_dict,os.path.join(rt_path,model_name,'train'))
        uploadToS3(validate_file_names,os.path.join(rt_path,model_name,'val'))
        print rt_path
        r = retrainModel(model_name,rt_path)
        wait_for_training(r)
        res = {i+1: runInferenceOnDict(test_file_names,model_name)}

        saveFileName = 'r{}.pickle'.format(i+1)
        pickleResults(output_path, saveFileName, [res,labeled_dict])
            



    hotdog = {
  "data": {
    "prediction": [
      {
        "label": "hotdog, hot dog, red hot", 
        "probability": 0.9803026914596558
      }, 
      {
        "label": "French loaf", 
        "probability": 0.005108409561216831
      }, 
      {
        "label": "ice lolly, lolly, lollipop, popsicle", 
        "probability": 0.0012415212113410234
      }, 
      {
        "label": "matchstick", 
        "probability": 0.0009466637275181711
      }, 
      {
        "label": "meat loaf, meatloaf", 
        "probability": 0.000607048103120178
      }
    ], 
    "success": "true"
  }
}

    # display benchmarks
    # pickle data

if __name__ == '__main__':
    main('tomato_potato')
        

    

def checkImages(images):
    for each_image in images:
        print each_image
        iamge_name = each_image.split('/')[-1]
        this_img = image.load_img(each_image, target_size = (299, 299))
        
        # image pre-processing
        x = np.expand_dims(image.img_to_array(this_img), axis=0)
        x = preprocess_input(x)
        x = x.copy(order="C")
        
        # encode
        x = base64_encode_image(x)
