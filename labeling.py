import boto3
import glob
import requests

class Image:
    def __init__(self, fname, label=None):
        self.file_name = fname
        self.label = label

    def location():n
        return self.file_name

    def getLabel():
        return self.label

    
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


def loadDirectory(path):
    class_paths = glob.glob(path + '/*')
    class_names = list(map(lambda x: x.split('/')[-1], class_paths))
    file_names = {x:glob.glob(path + x + '/*') for x in class_names}
    return class_names, file_names
    

def queryInferenceServer(fileName, model_name='base',
                         url='http://127.0.0.1:3031/inceptionV3/predict'):
    form_data = {'model_name' : model_name}
    files = {'image' : open(fileName, 'rb')}
    response = requests.post(url, files=files, data=form_data)
    return response.json()


def uploadToS3(file_dict, key_path, bucket_name='insightai2019'):
    # push all files in file_dict to S3
    s3 = boto3.client('s3')
    for key in file_dict:
        for datapoint in file_dict[key]:d
            file_key = key_path + datapoint[1:]
            s3.upload_file(datapoint, bucket_name, file_key)


def main(model_name, base_model='inceptionV3', N_initial=5,
         iterations=3, slices=5, bucket='insightai2019'):
    # shuffle/split

    # load the images - array of Images
    class_names, file_names = loadDirectory('./' + model_name + '/train/')
    validate_class_names, validate_file_names = loadDirectory(
        './' + model_name + '/val/')

    # first training pass takes N_initial of each class
    fn_train_dict = {}
    for key in file_names:
        fn_train_dict[key] = file_names[key][0:N_initial]
        del file_names[key][0:N_initial]
    uploadToS3(fn_train_dict,'models')
    
    # initialize benchmarks
    for i in range(iterations):
        # analyze the images - priority q?
        #setup bucket on aws
        #train
        #benchmarks
        pass

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
        

    


