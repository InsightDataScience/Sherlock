import subprocess as sp
from sys import argv
import requests

def main(fname):
    print "Hello {}".format(fname)
    url = 'http://127.0.0.1:3031/inceptionV3/predict'
    headers = {'Cache-Control' : 'no-cache',
               'Postman-Token' : 'eeedb319-2218-44b9-86eb-63a3a1f62e14',
               'content-type' : 'multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW'}
    files = {
        'image': ('hotdog.jpg', open('hotdog.jpg', 'rb')),
        'model_name': (None, 'base'),
       }
    response = requests.post(url, headers=headers, files=files)

    


if __name__ == '__main__':
    if len(argv) == 1:
        print "No arguements provided"
        pass;
    else:
        main(argv[1])
