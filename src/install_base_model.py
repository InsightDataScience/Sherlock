import os
import keras

BASE_MODEL_PATH = os.path.join("app", "models", "InceptionV3", "base", "base.h5")
TOPLESS_MODEL_PATH = os.path.join("app", "models", "InceptionV3", "topless")

# loading base model
if os.path.exists(BASE_MODEL_PATH):
    print "* Starting: Found Base Model."
else:
    print "* Starting: No Base Model Found. Loading..."
    base_model = keras.applications.inception_v3.InceptionV3(include_top=True, 
                                                    weights='imagenet',
                                                    input_shape=(299, 299, 3))
    base_model.save(BASE_MODEL_PATH)
    print "* Starting: Base Model Saved!"
    
# loading topless model
if os.path.exists(TOPLESS_MODEL_PATH):
    print "* Starting: Found Topless Model."
else:
    os.makedirs(TOPLESS_MODEL_PATH)
    print "* Starting: No Topless Model Found. Loading..."
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, 
                                                    weights='imagenet',
                                                    input_shape=(299, 299, 3))
    base_model.save(os.path.join(TOPLESS_MODEL_PATH, "topless.h5"))
    print "* Starting: Topless Model Saved!"