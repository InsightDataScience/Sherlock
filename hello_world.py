from sys import argv
import requests

def main(fname):
    
    print "Hello {}".format(fname)
    url = 'http://127.0.0.1:3031/inceptionV3/predict'
    headers = {
        'Cache-Control':'no-cache',
        'Postman-Token': 'eeedb319-2218-44b9-86eb-63a3a1f62e14',
        'content-type': 'multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW'
    }

    form_data = {
        [unicode('model_name'): unicode('base'),
        'image': open('hotdog.jpg', 'rb')]
    }
    # form_data = {'image':'@hotdog.jpg', 'model_name':'base'}

    try:
#       response = requests.post(url, headers=headers, files=files,stream=True)
        response = requests.post(url, headers=headers, data=form_data)
        print response.raw
    except requests.exceptions.RequestException as req_err:
        print req_err



if __name__ == '__main__':
    if len(argv) == 1:
        print "No arguements provided"
    else:
        main(argv[1])
