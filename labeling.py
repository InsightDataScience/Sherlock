import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics import pairwise_distances_argmin_min




def labeling_priority(data_unlabeled, trained_model,n,method='ed',data_labeled=None):
    if method == 'random':
        return randomly_choose_n(data_unlabeled, n)
    if method == 'ed':
        unlabeled_features = feature_extraction(data_unlabeled, trained_model)
        points = pick_points_faster(unlabeled_features, trained_model, n)
        labeled_files = [data_unlabeled[idx] for idx in points]
        return labeled_files


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








def choose_n(file_dict, n):
#selects n from each class 
    ret_dict = {}
    for class_name in file_dict:
        if file_dict[class_name]:
            ret_dict[class_name] = file_dict[class_name][:n]                         
    return ret_dict


def randomly_choose_n(file_list, n):
#randomly selects n total files
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
