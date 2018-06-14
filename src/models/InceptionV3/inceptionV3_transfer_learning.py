'''
Created on Jun 13, 2018

@author: runshengsong
'''
import os
import glob

import keras

from keras.optimizers import SGD
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

class InceptionTransferLeaner:
    def __init__(self, model_name):
        try:
            self.topless_model = load_model('./topless/topless.h5')
        except IOError:
            # load model from keras
            self.topless_model = InceptionV3(include_top=False, 
                                            weights='imagenet',
                                            input_shape=(299, 299, 3))

    def transfer_model(self, local_dir,
                       nb_epoch,
                       batch_size):
        """
        transfer the topless InceptionV3 model
        to classify new classes
        """
        train_dir = os.path.join(local_dir, "train")
        val_dir = os.path.join(local_dir, "val")
        
        # set up parameters
        nb_train_samples = self.__get_nb_files(train_dir)
        nb_classes = len(glob.glob(train_dir + "/*"))
        nb_val_samples = self.__get_nb_files(val_dir)
        nb_epoch = int(nb_epoch)
        batch_size = int(batch_size)
        
        # data prep
        train_datagen =  ImageDataGenerator(
            preprocessing_function = preprocess_input
          )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
            )
        
        # generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(299, 299),
            batch_size=batch_size)
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(299, 299),
            batch_size=batch_size,
            )
        
        # set up transfer learning model
        self.topless_model = self.
        
    
    def __add_new_last_layer(self, topless_model, nb_class):
        """
        add the last layer to the topless model
        """
        x = topless_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
        predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
        model = Model(input=base_model.input, output=predictions)
        return model
        
    
    def __get_nb_files(self, directory):
        """Get number of files by searching local dir recursively"""
        if not os.path.exists(directory):
            return 0
        cnt = 0
        for r, dirs, files in os.walk(directory):
            for dr in dirs:
                cnt += len(glob.glob(os.path.join(r, dr + "/*")))
        return cnt
