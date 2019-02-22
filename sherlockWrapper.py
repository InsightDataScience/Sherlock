def query_inference_server(fileName, model_name='base',
                         url='http://127.0.0.1:3031/inceptionV3/predict'):
    form_data = {'model_name': model_name}
    files = {'image': open(fileName, 'rb')}
    response = requests.post(url, files=files, data=form_data)
    return response.json()


def train_new_model(model_name, bucket_name='insightai2019', path_prefix='models',
                         url='http://127.0.0.1:3031/inceptionV3/transfer'):
    form_data = {
        'train_bucket_name': bucket_name, 
        'train_bucket_prefix': os.path.join(path_prefix, model_name)
    }

    response = requests.post(url, data=form_data)
    return response.json()


def check_status(id, url='http://127.0.0.1:3031/tasks/info'):
    response = requests.post(url,json={id:id})
    return response.json()['Tasks Status'][0]['status'] == 'SUCCESS'


def retrain_model(model_name, path, bucket_name='insightai2019',
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


def upload_to_S3(file_dict, key_path, bucket_name='insightai2019'):
    # push all files in file_dict to S3
    s3 = boto3.client('s3')
    for key in file_dict:
        for datapoint in file_dict[key]:
            fname= os.path.split(datapoint)[-1]
            file_key = os.path.join(key_path,key,fname)
            print file_key
            s3.upload_file(datapoint, bucket_name, file_key)


def wait_for_training(response, t=20, t_max=900,
                      url='http://127.0.0.1:3031/tasks/info'):
    status = check_status(response['task_id'],url)
    while not status:
        time.sleep(t)
        t += t / 10
        status = check_status(response['task_id'],url)
    return 1
