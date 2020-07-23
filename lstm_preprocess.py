import os
import collections

import tensorflow as tf
from tokenization import FastaTokenizer
from absl import logging

import argparse



# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))

def _float_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

def _int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


class TrainingInstance(object):
    def __init__(self, input_ids):
        features = collections.OrderedDict()
        features['input_ids'] = _int64_feature(input_ids)

        self.tf_example = tf.train.Example(features=tf.train.Features(feature=features))



    def serialize(self):
        return self.tf_example.SerializeToString()



class TrainingExmpleWriter():
    """
    Data writer. It will round-robin write to the output file paths in TFRecord format
    Record will be stored in buffer until threshold is reached before writing 
    for performance on large files
    Buffer Size is number of records

    output_files is a comma seperated list of file paths

    **IMPORTANT**
    Must call flush_and_close() when finished to make sure all data is written to disk
    """
    def __init__(self, output_files: str):
        self.writers = self._create_output_writers(output_files)
        self.writer_index = 0
        self.total_written = 0


    def flush_and_close(self):
        for writer in self.writers:
            writer.flush()
            writer.close()


    def write(self, trainingInstance: TrainingInstance):
        self.writers[self.writer_index].write(trainingInstance.serialize())

        self.writer_index = (self.writer_index + 1) % len(self.writers)

        self.total_written += 1


    def _create_output_writers(self, output_paths: str):
        output_files = [f for f in output_paths.split(",") if f]

        output_file_writers = []
        logging.info("*** Writing output files ***")
        for output_file in output_files:
            logging.info("  %s", output_file)
            ouput_writer = tf.io.TFRecordWriter(output_file)
            output_file_writers.append(ouput_writer)

        return output_file_writers


def process_input_file(input_file, tokenizer: FastaTokenizer,
                        dataWriter:TrainingExmpleWriter, max_seq_length):
    with tf.io.gfile.GFile(input_file, 'r') as reader:
        i = 0
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()

            tokens = tokenizer.tokenize(line)
            token_ids = tokenizer.tokens_to_ids(tokens)

            if len(token_ids) > max_seq_length:
                token_ids = token_ids[:max_seq_length]
            elif len(token_ids) < max_seq_length:
                diff = max_seq_length - len(token_ids)
                padding = [0] * diff
                token_ids += padding

            assert len(token_ids) == max_seq_length

            training_instance = TrainingInstance(token_ids)

            dataWriter.write(training_instance)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_files", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_seq_len", default=512)
    parser.add_argument("--vocab_file", required=True)
    args = parser.parse_args()

    tokenizer = FastaTokenizer(vocab_file=args.vocab_file.strip())
    tokenizer.load_vocab()

    input_files = []
    for input_pattern in args.input_files.split(","):
        input_pattern = input_pattern.strip()
        logging.info("*** Reading from input files ***")
        logging.info(" %s ", input_pattern)
        input_files.extend(tf.io.gfile.glob(input_pattern))

    writer = TrainingExmpleWriter(args.output_path) 

    logging.info("*** Reading from input files ***")
    for input_file in input_files:
        logging.info("  %s", input_files)

    for file_path in input_files:        
        process_input_file(input_pattern, tokenizer, writer, args.max_seq_len)


    writer.flush_and_close()
    logging.info("Total training examples written: %d", writer.total_written)