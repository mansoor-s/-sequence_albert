# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
# coding=utf-8
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import collections
import six
import tensorflow as tf


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return six.ensure_text(text, "utf-8", "ignore")
  else:
    raise ValueError("Unsupported string type: %s" % (type(text)))


class FastaTokenizer(object):
  """Runs basic FASTA tokenization (charecter tokenizer)."""

  def __init__(self, vocab_file, do_lower_case=False, mask_token="[MASK]", unknown_token="X"):
    """Constructs a FastaTokenizer.
    `mask_token` is ignored if loading a vocab file 

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.mask_token = mask_token
    self.unknown_token = unknown_token
    self.do_lower_case = do_lower_case
    self.vocab_file = vocab_file
    self.vocab = {}
    self.inv_vocab = {}

  def vocab_words(self):
    return list(self.vocab.keys())

  def vocab_size(self):
    return len(self.vocab)


  def load_vocab(self):
    """Loads a vocabulary file into a dictionary."""

    # Add empty NULL token:
    self.inv_vocab[0] = ''
    self.vocab[''] = 0

    line_num = 0
    with tf.io.gfile.GFile(self.vocab_file, "r") as reader:
      while True:
        token = reader.readline()
        if not token:
          break
        token = token.strip().split()[0] if token.strip() else ""
        if not token:
          logging.info("Malformed line in vocab file: %i", len(line_num))
          continue

        if token not in self.vocab:
          self.inv_vocab[len(self.vocab)] = token
          self.vocab[token] = len(self.vocab)

        line_num += 1
    

  def tokenize(self, text) -> [str]:
    """Tokenizes a piece of text."""
    assert len(self.vocab)

    tokens = [c for c in text]
    return tokens


  def tokens_to_ids(self, tokens: [str]) -> [int]:
    return [self.vocab[t] if t in self.vocab else self.vocab[self.unknown_token] for t in tokens]


  def ids_to_tokens(self, ids: [int]) -> [str]:
    return [self.inv_vocab[i] for i in ids]