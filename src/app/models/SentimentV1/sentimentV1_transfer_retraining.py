'''
Created on Jan 13, 2018

Most of the code is taken from run_classifier.py from
the Google BERT github - https://github.com/google-research/bert
'''
import os
import glob
import logging
import boto3

#Import BERT related file - USED as is from Google/BERT
import modeling
import optimization
import run_classifier
import tokenization
import tensorflow as tf
import csv

import settings
import datetime

from app import app

INCEPTIONV3_TOPLESS_MODEL_PATH = app.config['INCEPTIONV3_TOPLESS_MODEL_PATH']
BERT_MODEL_PATH = app.config['BERT_MODEL_PATH']


class BertTransferLeaner:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def getlabel(self,data_dir):
        with tf.gfile.Open(os.path.join(data_dir, "train.tsv"),"r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            label = []
            for line in reader:
                line_label = line[1]
                if line_label not in label:
                    label.append(line_label)
        return label


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
        BERT_PRETRAINED_DIR = BERT_MODEL_PATH
        OUTPUT_DIR = os.path.join(local_dir,'output_bert')
        DATA_DIR = os.path.join(local_dir,'data')
        logging.info('***** Model output directory: %s*****',OUTPUT_DIR)
        logging.info('***** BERT pretrained directory: %s *****',BERT_PRETRAINED_DIR)
        logging.info('***** DATA directory: %s *****',DATA_DIR)
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

        logging.info("Found VOCAB File:%s",VOCAB_FILE)
        bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
        tf.gfile.MakeDirs(OUTPUT_DIR)
        processor = run_classifier.ColaProcessor()
        #label_list = processor.get_labels()
        label_list = self.getlabel(DATA_DIR)
        print label_list
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
        final_ckpt = estimator.latest_checkpoint()
        print('***** Finished training at {} *****'.format(datetime.datetime.now()))
        logging.info("*****Final Checkpoint*****%s",final_ckpt)
        final_ckpt_file = os.path.join(OUTPUT_DIR, "final_ckpt.txt")
        with tf.gfile.GFile(final_ckpt_file, "w") as writer:
            writer.write("%s" % final_ckpt)


        # Do Eval
        logging.info('Starting Eval..')
        eval_examples = processor.get_dev_examples(DATA_DIR)
        label_list = self.getlabel(DATA_DIR)
        print label_list
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(OUTPUT_DIR, "eval.tf_record")
        run_classifier.file_based_convert_examples_to_features(
            eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
        eval_steps = None

        eval_input_fn = run_classifier.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=MAX_SEQ_LENGTH,
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
    
    def test_model(self, local_dir,
                       nb_epoch,
                       batch_size,
                       bucket_name):
        """
        Use the BERT Uncased language model to train on
        new data
        """
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("*:BERT MODEL PATH:%s",BERT_MODEL_PATH)
        logging.info("*:Local Dir%s",local_dir)


        mod_name = self.model_name
        BERT_MODEL = 'uncased_L-12_H-768_A-12'
        BERT_PRETRAINED_DIR = BERT_MODEL_PATH
        OUTPUT_DIR = os.path.join(local_dir,'output_bert')
        DATA_DIR = os.path.join(local_dir,'data')
        logging.info('***** Model output directory: %s*****',OUTPUT_DIR)
        logging.info('***** BERT pretrained directory: %s *****',BERT_PRETRAINED_DIR)
        logging.info('***** DATA directory: %s *****',DATA_DIR)
        TRAIN_BATCH_SIZE = 32
        EVAL_BATCH_SIZE = 8
        PREDICT_BATCH_SIZE = 32
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
        with open(os.path.join(OUTPUT_DIR,'final_ckpt.txt')) as f:
            content = f.readlines()
            logging.info("***Final_cktp->%s\n",content)
        test_ckpt = content[0].split('/')[-1]
        INIT_CHECKPOINT = os.path.join(OUTPUT_DIR, test_ckpt)
        DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

        logging.info("Found VOCAB File:%s",VOCAB_FILE)
        bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
        tf.gfile.MakeDirs(OUTPUT_DIR)
        processor = run_classifier.ColaProcessor()
        #label_list = processor.get_labels()
        label_list = self.getlabel()
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
            eval_batch_size=EVAL_BATCH_SIZE,
            predict_batch_size=PREDICT_BATCH_SIZE)
        
        predict_examples = processor.get_test_examples(DATA_DIR)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(OUTPUT_DIR, "predict.tf_record")
        run_classifier.file_based_convert_examples_to_features(predict_examples, label_list,
                                                MAX_SEQ_LENGTH, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", batch_size)

        predict_input_fn = run_classifier.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(OUTPUT_DIR, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        s3 = boto3.resource('s3')
        tf.logging.info("Done with prediction uploading results to S3")
        try:
            s3.Bucket(bucket_name).upload_file(output_predict_file, output_predict_file)
        except Exception as err:
            logging.info("Unable to upload to S3")
            logging.info(err)


        return 1
    
