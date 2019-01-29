'''
Created on Jan 13, 2018

@author: manu
'''
import os
import glob
import logging

import settings

from app import app

INCEPTIONV3_TOPLESS_MODEL_PATH = app.config['INCEPTIONV3_TOPLESS_MODEL_PATH']



class BertTransferLeaner:
    def __init__(self, model_name):
        self.model_name = model_name
        # the creation of the model directory should be handled
        # in the API
        try:
            logging.info("* Transfer: Loading Topless Model...")
            self.topless_model = load_model(INCEPTIONV3_TOPLESS_MODEL_PATH)

        except IOError:
            # load model from keras
            print "* Transfer: Loading Topless Model from Keras..."
            self.topless_model = InceptionV3(include_top=False,
                                            weights='imagenet',
                                            input_shape=(299, 299, 3))

        self.new_model = None # init the new model

    def transfer_model(self, local_dir,
                       nb_epoch,
                       batch_size):
        """
        transfer the topless InceptionV3 model
        to classify new classes
        """

        BERT_MODEL = 'uncased_L-12_H-768_A-12'
        BERT_PRETRAINED_DIR = f'{repo}/uncased_L-12_H-768_A-12'
        OUTPUT_DIR = f'{repo}/outputs'
        print(f'***** Model output directory: {OUTPUT_DIR} *****')
        print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')

        train_dir = os.path.join(local_dir, "train")
        val_dir = os.path.join(local_dir, "val")



        VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
        CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
        INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
        DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

        label_list = ['0', '1']
        tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
        train_examples = create_examples(train_lines, 'train', labels=train_labels)

        tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver
#       TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
        run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=OUTPUT_DIR,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        num_train_steps = int(
            len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        model_fn = run_classifier.model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
            num_labels=len(label_list),
            init_checkpoint=INIT_CHECKPOINT,
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available
            use_one_hot_embeddings=True)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available
            model_fn=model_fn,
            config=run_config,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE)
        # return the model
        
    def __setup_to_finetune(self, model, nb_layer_to_freeze):
        """
        Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
        Args:
        model: keras model
        """
        for layer in model.layers[:nb_layer_to_freeze]:
            layer.trainable = False
        for layer in model.layers[nb_layer_to_freeze:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    def __setup_to_transfer_learn(self, model, base_model):
        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def __add_new_last_layer(self, topless_model, nb_classes):
        """
        add the last layer to the topless model
        """
        x = topless_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(settings.FC_SIZE, activation='relu')(x) #new FC layer, random init
        predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
        model = Model(input=topless_model.input, output=predictions)
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
