curl -X POST \
  http://127.0.0.1:3031/inceptionV3/transfer \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: 4e90e1d6-de18-4501-a82c-f8a878616b12' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F train_bucket_name=insightai2019 \
  -F train_bucket_prefix=models/tomato_potato


