'''
Created on Jan 13, 2018

@author: manu
'''
import os
import glob
import logging
import tensorflow as tf
#Import BERT related file - USED as is from Google/BERT
import bert.modeling
import bert.optimization
import bert.run_classifier
import bert.tokenization

import settings
import datetime

from app import app

INCEPTIONV3_TOPLESS_MODEL_PATH = app.config['INCEPTIONV3_TOPLESS_MODEL_PATH']
BERT_MODEL_PATH = app.config['BERT_MODEL_PATH']


class BertTransferLeaner:
    def __init__(self, model_name):
        self.model_name = model_name

    def traineval_model(self, local_dir,
                       nb_epoch,
                       batch_size):
        """
        Use the BERT Uncased language model to train on
        new data
        """
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("*:BERT MODEL PATH:%s",BERT_MODEL_PATH)
        logging.info("*:Local Dir%s",local_dir)


        mod_name = self.model_name
        BERT_MODEL = 'uncased_L-12_H-768_A-12'
        BERT_PRETRAINED_DIR = os.path.join(BERT_MODEL_PATH,'uncased_L-12_H-768_A-12')
        OUTPUT_DIR = os.path.join(local_dir,'output_bert')
        DATA_DIR = os.path.join(local_dir,'data')
        logging.info('***** Model output directory: %s*****',OUTPUT_DIR)
        logging.info('***** BERT pretrained directory: %s *****',BERT_PRETRAINED_DIR)

        TRAIN_BATCH_SIZE = 32
        EVAL_BATCH_SIZE = 8
        LEARNING_RATE = 2e-5
        NUM_TRAIN_EPOCHS = 3.0
        WARMUP_PROPORTION = 0.1
        MAX_SEQ_LENGTH = 128
        # Model configs
        # if you wish to finetune a model on a larger dataset, use larger interval
        SAVE_CHECKPOINTS_STEPS = 1000
        # each checpoint weights about 1,5gb
        ITERATIONS_PER_LOOP = 1000
        NUM_TPU_CORES = 8

        VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR,'vocab.txt')
        BERT_CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR,'bert_config.json')
        INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
        DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

        bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
        tf.gfile.MakeDirs(OUTPUT_DIR)
        processor = run_classifier.ColaProcessor()
        label_list = processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

        # Since training will happen on GPU, we won't need a cluster resolver
        tpu_cluster_resolver = None
        # TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=OUTPUT_DIR,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=ITERATIONS_PER_LOOP,
                num_shards=NUM_TPU_CORES,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None
        train_examples = processor.get_train_examples(DATA_DIR)
        num_train_steps = int(
            len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        model_fn = run_classifier.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=INIT_CHECKPOINT,
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
            use_one_hot_embeddings=False) #Try with True

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
            model_fn=model_fn,
            config=run_config,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE)

        # Train the model.
        logging.info('Starting Training...')
        train_file = os.path.join(OUTPUT_DIR, "train.tf_record")
        run_classifier.file_based_convert_examples_to_features(
            train_examples, label_list, MAX_SEQ_LENGTH, tokenizer, train_file)
        tf.logging.info('***** Started training at {} *****'.format(datetime.datetime.now()))
        tf.logging.info('  Num examples = {}'.format(len(train_examples)))
        tf.logging.info('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = run_classifier.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))
        # Do Eval
        logging.info('Starting Eval..')
        eval_examples = processor.get_dev_examples(DATA_DIR)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(OUTPUT_DIR, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

        return result

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
