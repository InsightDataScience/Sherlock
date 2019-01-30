curl -X POST \
  http://127.0.0.1:3031/inceptionV3/predict \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: eeedb319-2218-44b9-86eb-63a3a1f62e14' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F image=@$1 \
  -F model_name=$2
