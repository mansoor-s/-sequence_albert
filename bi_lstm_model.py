import tempfile
import argparse

import tensorflow as tf
from tensorflow.python.lib.io import file_io

import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation
from tensorflow.keras.layers import Embedding, RepeatVector, Reshape

import numpy as np
from absl import logging

from tokenization import FastaTokenizer


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'input_ids': tf.io.VarLenFeature(tf.int64)}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    dense_input_ids = tf.sparse.to_dense(parsed_features['input_ids'])
    
    return dense_input_ids, dense_input_ids

def create_dataset(file_paths, batch_size, num_parallel_reads=8, buffer_size=1e+8):
    """
        Create a dataset from multiple TFRecord files in parrallel reads
        Buffer size is 100MB
    """

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=8, 
                                        buffer_size=buffer_size)
    
    # Maps the parser on every filepath in the array. 
    # You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    #Data is pre-shuffled
    # Set the number of datapoints you want to load and shuffle 
    #dataset = dataset.shuffle(SHUFFLE_BUFFER)
    
    # Set the batchsize
    batched_dataset = dataset.batch(batch_size)
    
    return batched_dataset


def get_new_model(sequence_length, vocab_size):
    model = Sequential()

    model.add(Embedding(vocab_size, 128))
    model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(852, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(852, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(1024, activation='relu', return_sequences=False)))
    model.add(Dense(sequence_length, activation="softmax"))

    return model


def create_tpu_strategy(tpu_name):
    tpu_worker = 'grpc://{}'.format(tpu_name.strip())
    logging.info('Using TPU worker: %s', tpu_worker)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_worker)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    return strategy


def download_model_checkpoint(path):
    logging.info('Downloading model checkpoint: ', path)
    model_file = file_io.FileIO(path, mode='rb')
    file_name = path.split('/')[-1:]
    downloaded_path = '{}/{}'.format(tempfile.gettempdir(), file_name)
    temp_downloaded_model = open(downloaded_path, 'wb')
    temp_downloaded_model.write(model_file.read())
    temp_downloaded_model.close()

    return keras.models.load_model(downloaded_path)



def make_or_restore_model(sequence_length, vocab_size, checkpoint_path):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = tf.io.gfile.glob(checkpoint_path)
    if checkpoints:
        checkpoint_files = [f for f in checkpoints if 'ckpt-' in f]
        if checkpoint_files:
            stats = [tf.io.stat(f) for f in checkpoint_files]
            checkpoint_file_stats = list(zip(checkpoint_files, stats))
            checkpoint_file_stats.sort(key=lambda stat: stat[1], reverse=True)

            latest_checkpoint = checkpoint_file_stats[0]
            logging.info("Restoring from", latest_checkpoint)
            return download_model_checkpoint(latest_checkpoint)
   
    logging.info("Creating a new model")
    return get_new_model(sequence_length, vocab_size)



def start_pretraining(training_path, tpu_name, sequence_length,
                        batch_size, vocab_size, num_epochs, checkpoint_path, model_log_path):
    """
        batch_size must be a multiple of number of TPU cores (8)
    """

    if batch_size % 8:
        raise 'batch_size must be a multiple of number of TPU cores: 8'

    #tpu_strategy = create_tpu_strategy(tpu_name)

    logging.info('Creating/Loading model')
    #with tpu_strategy.scope():
    model = make_or_restore_model(sequence_length, vocab_size, checkpoint_path)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    
    logging.info('Gather training and validation data files')
    input_files = tf.io.gfile.glob(data_source)
    validation_files = input_files[-1:]
    training_files = input_files[:-1]

    logging.info('Training files: ', training_files)
    logging.info('Validation files: ', validation_files)

    logging.info('Creating training and validation datasets')
    training_dataset = create_dataset(input_files, batch_size)
    validation_dataset = create_dataset(evaluation_file, batch_size)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + "ckpt-{epoch}", save_freq=1000
        ),
        keras.callbacks.TensorBoard(log_dir=model_log_path, update_freq=1000)
    ]

    logging.info('Starting model fit')
    model.fit(epochs=num_epochs, steps_per_epoch=batch_size, validation_data=validation_dataset)
    logging.info('Model fit finished')



if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_path", required=True)
    #parser.add_argument("--validation_path", required=True)
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--sequence_length", default=512)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--num_epochs", required=True)
    parser.add_argument("--tpu_name", required=True)
    parser.add_argument("--model_log_path", required=True)
    
    args = parser.parse_args()
    logging.info("Arguments recieved: ")
    logging.info('  %s', args)

    tokenizer = FastaTokenizer(args.vocab_file)
    vocab_size = tokenizer.load_vocab()

    start_pretraining(args.training_path, args.tpu_name,
                        args.sequence_length, int(args.batch_size), vocab_size,
                        args.num_epochs, args.model_save_path, args.model_log_path)