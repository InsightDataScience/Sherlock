# Sherlock
Sherlock is a web platform that allows user to create a image classifier for custom images, based on pre-trained CNN models. It also allows to use the customized CNN to pre-label images, and re-train the customized CNN when more training data become avaliable.

Sherlock is currently serving as RESTful APIs.

- [Sherlock for NLP](#sherlock-for-nlp)


[Here](http://bit.ly/michaniki_demo) are the slides for project Sherlock (previously called Michaniki).

---
## Development
Everything is wrapped in Docker, here are the core packages used in *Sherlock*.
* Python 2.7.3
* uWSGI
* Flask
* Docker (built on 17.12.0-ce)
* Docker Compose (1.18.0)
* Keras + Tensorflow

## Transfer Learning Explained
*Sherlock* does transfer learning and fine-tuning (two steps) on pre-trained deep CNNs on a custom image dataset. If you want to setup *Sherlock*, skip to the next section.

## Setup environment
#### 1. Install Docker:
On Linux Machines (Ubuntu/Debian) -- UNTESTED!

```bash
sudo apt-cache policy docker-ce
sudo apt-get install docker-ce=17.12.0-ce
```

On Mac :
Download [Docker from here](https://store.docker.com/editions/community/docker-ce-desktop-mac) and install it.

#### 2. Install Docker Compose
Docker compose can start multiple containers at the same time.
On Linux Machines (Ubuntu/Debian)  -- UNTESTED!:

```bash
sudo curl -L https://github.com/docker/compose/releases/download/1.18.0/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

On Mac:
The package of Docker-CE for Mac already come with docker compose.

#### 3. Install Git:
[Install git](https://git-scm.com/downloads) for the appropriate system, if you don't have yet.

## Build and Start *Sherlock*
#### 1. Clone the repo:

```bash
git clone https://github.com/InsightDataCommunity/Sherlock
```

#### 2. Export Your AWS Credentials to Host Environment.
*Sherlock* needs to access S3 for customized images. You should create an IAM user at [AWS](https://aws.amazon.com/), if you don't yet have an AWS account, create one.

Then, export your credentials to your local environment:
```bash
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_ID
export AWS_SECRET_ACCESS_KEY=YOUR_ACCESS_KEY
```

You now should be able to see your credentials by:

```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

Docker will access these environment variables and load them to docker.

#### 3. Start the Docker Containers:
Move to the directory where you cloned *Sherlock* , and run:
```bash
docker-compose up --build
```

If everything goes well, you should start seeing the building message of the docker containers:
```
Building michaniki_client
Step 1/9 : FROM continuumio/miniconda:4.4.10
 ---> 531588d20a85
Step 2/9 : RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
 ---> Using cache
 ---> a356e16a75e7
...
```

The first time building might take few minutes. Once finished, you should see the *uWSGI* server, *inference* server running, something like this:
```
...
michaniki_client_1  | *** uWSGI is running in multiple interpreter mode ***
michaniki_client_1  | spawned uWSGI master process (pid: 8)
michaniki_client_1  | spawned uWSGI worker 1 (pid: 18, cores: 2)
michaniki_client_1  | spawned uWSGI worker 2 (pid: 19, cores: 2)
michaniki_client_1  | spawned uWSGI worker 3 (pid: 21, cores: 2)
michaniki_client_1  | spawned uWSGI worker 4 (pid: 24, cores: 2)
```

Then *sherlock* is ready for you.

## Use *Sherlock*:
*Sherlock* currently provides 3 major APIs. To test *Sherlock*, I recommend testing the APIs using [POSTMAN](https://www.getpostman.com/). The examples below are in terminal, using `cURL`

#### 1. Welcome Page
If *Sherlock* is running correctly, go to `http://127.0.0.1:3031/` in your web browser, you should see the welcome message of *Sherlock*.

#### 2. Predict a Image with InceptionV3
*Sherlock* can classify any image to ImageNet labels using a pre-trained InceptionV3 CNN:

```bash
curl -X POST \
  http://127.0.0.1:3031/inceptionV3/predict \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: eeedb319-2218-44b9-86eb-63a3a1f62e14' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F image=@*PATH/TO/YOUR/IMAGE* \
  -F model_name=base
```

replace the name after ``` image=@ ``` with the path of the image you want to run. *Sherlock* will return the image labels and probabilities, ranked from high to low, in `json` format:

The model name *base* is used to refer to the basic InceptionV3 model.

```
{
    "data": {
        "prediction": [
            {
                "label": "Egyptian cat",
                "probability": 0.17585527896881104
            },
            {
                "label": "doormat, welcome mat",
                "probability": 0.057334817945957184
            },
			...
```

#### 3. Create a New Model Using Custom Image Dataset:
*Sherlock* can do transfer learning on the pre-trained InceptionV3 CNN (without the top layer), and create a new CNN for users' image dataset.

**The new image dataset should be stored at S3 first, with the directory architecture in S3 should look like this**:
```
.
├── YOUR_BUCKET_NAME
│   ├── models
│       ├── YOUR_MODEL_NAME
│   	        ├── train
│   		    ├── class1
│   		    ├── class2
│   	        ├── val
│   		    ├── class1
│   		    ├── class2
```

The folder name you give to *YOUR_MODEL_NAME* will be used to identify this model once it is trained.

The name of train and val folders **can't be changed**. The folder names for different classes will be used as the label of the class, you can create as many class folders as you want.

**The S3 folders should have public access permission**.

To call this API, do:
```bash
curl -X POST \
  http://127.0.0.1:3031/inceptionV3/transfer \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: 4e90e1d6-de18-4501-a82c-f8a878616b12' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F train_bucket_name=YOUR_BUCKET_NAME \
  -F train_bucket_prefix=models/YOUR_MODEL_NAME
```
*Sherlock* will use the provided images to create a new model to classify the classes you provided in the S3 folder. Once transfer learning is done, you can use the new model to label images by pass **YOUR_MODEL_NAME** to the inference API described earlier.

#### 4. Labeling new images:

Once transfer learning is finished, you can use the newly created model to label your new images. The new image folder should also be hosted in S3. The structure is pretty similar to the one used in the transfer learning API. Please structure your folder like this:

```
.
├── YOUR_BUCKET_NAME
│   ├── YOUR_IMAGE_FOLDER
│   	├── img1
|		├── img2
|		├── img3
|		├── img4
|		├── img5
|		├── img6
...
```

Then, you can call the API like this:

```bash
curl -X POST \
  http://127.0.0.1:3031/inceptionV3/label \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: bf736848-d455-4c6c-9ec4-38a047e05e15' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F s3_bucket_name=S3_BUCKET_NAME \
  -F s3_bucket_prefix=S3_BUCEKT_PREFIX \
  -F model_name=sharkVsalmon
```

The return will be something like this:
```
{
    "data": [
        {
            "image name": "salmon4.JPEG",
            "prediction": [
                {
                    "label": "shark",
                    "probability": 1
                },
                {
                    "label": "salmon",
                    "probability": 0
                }
            ]
        },
        {
            "image name": "shark1.JPEG",
            "prediction": [
                {
                    "label": "shark",
                    "probability": 0.998984158039093
                },
                {
                    "label": "salmon",
                    "probability": 0.001015781774185598
                }
            ]
        },
...
```


#### 5. Resume training on existing model:

Once more labeled images become available, you can retrain exiting models by submitting additional model folders. The structure should be the same as the one used by the transfer learning API.

```bash
curl -X POST \
  http://127.0.0.1:3031/inceptionV3/retrain \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: 0710ad89-a997-423f-a11f-1708df195dad' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F nb_epoch=3 \
  -F batch_size=2 \
  -F train_bucket_name=S3_BUCKET_NAME \
  -F train_bucket_prefix=S3_BUCEKT_PREFIX/sha
```

## Sherlock for NLP

### 1. Predict Sentiment of Text with Simple run_classifier
```bash
curl -X POST \
  http://127.0.0.1:3031/sentimentV1/predict \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: eeedb319-2218-44b9-86eb-63a3a1f62e14' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F textv='the movie was bad' \
  -F model_name=base
```

### 2. Train a new classification model using pre-trained BERT model

**The new text dataset should be stored at S3 first, with the directory architecture in S3 should look like this**:
```
.
├── YOUR_BUCKET_NAME
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv		
```
The folder name you give to *YOUR_MODEL_NAME* will be used to identify this model once it get trained.

The name of train, dev and test files  **can't be changed**.
The train and dev file should have below format (without header)-
id label None Sentence
1  0     NC   
The test.tsv file should only have id and sentence column (with header)
**The S3 folders should have public access permission**.

To call this API, do:
```bash
curl -X POST \
  http://127.0.0.1:3031/sentimentV1/trainbert \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: 4e90e1d6-de18-4501-a82c-f8a878616b12' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F train_bucket_name=YOUR_BUCKET_NAME \
  -F train_bucket_prefix=YOUR_MODEL_NAME
```