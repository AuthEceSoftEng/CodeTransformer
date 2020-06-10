from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import os
import re
import pickle
import math as m
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from model import *

# loading the docstring vocabulary.
docstring_vocab =  pickle.load(open('data/docstring_vocab.pkl', 'rb'))
# loading the code vocabulary.
code_vocab =  pickle.load(open('data/code_vocab.pkl', 'rb'))

# loading the functions, function tokens and URLs of the whole corpus.
functions = pd.read_pickle('data/functions.pkl')

# copying corpus' code_tokens column.
corpus_function_tokens = functions['function_tokens'].copy(deep=True)

def encode(inp, tar, input_encoder, target_encoder):
  # encoding input data.
  for index, row in enumerate(inp):
    inp[index] = [input_encoder.encode(token)[0] for token in row]

  # encoding target data.
  for index, row in enumerate(tar):
    tar[index] = [target_encoder.encode(token)[0] for token in row]

  return inp, tar

def to_numpy(inp, tar):
  # converting input data to numpy.
  inp = pd.DataFrame(list(inp))
  inp = inp.to_numpy()
  inp = np.nan_to_num(inp)
  inp = inp.astype(int)

  # converting target data to numpy.
  tar = pd.DataFrame(list(tar))
  tar = tar.to_numpy()
  tar = np.nan_to_num(tar)
  tar = tar.astype(int)

  return inp, tar

# building input_encoder.
input_encoder = tfds.features.text.TokenTextEncoder(docstring_vocab)
# building target_encoder.
target_encoder = tfds.features.text.TokenTextEncoder(function_vocab)

# applying encoding to the whole corpus and converting it to numpy.
_, encoded_corpus_target = encode([[]], corpus_function_tokens, input_encoder, target_encoder)
_, encoded_corpus_target = to_numpy([[]], encoded_corpus_target)

BUFFER_SIZE = 500000

# creating a tensor dataset with the whole corpus.
corpus_dataset = tf.data.Dataset.from_tensor_slices(encoded_corpus_target)
# caching the dataset for performance optimizations.
corpus_dataset = corpus_dataset.cache()
corpus_dataset = corpus_dataset.batch(1000)
corpus_dataset = corpus_dataset.prefetch(tf.data.experimental.AUTOTUNE)

NUM_LAYERS = 3
INPUT_VOCAB_SIZE = input_encoder.vocab_size
TARGET_VOCAB_SIZE = target_encoder.vocab_size
INPUT_POSITION = input_encoder.vocab_size
TARGET_POSITION = target_encoder.vocab_size
NUM_HEADS = 8
DFF = 512
D_MODEL = 128
RATE = 0.1

matching_network = MatchingNetwork(NUM_LAYERS, INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                                   INPUT_POSITION, TARGET_POSITION, NUM_HEADS,
                                   DFF, D_MODEL, RATE)

if os.path.isfile('models/weights.index'):
  matching_network.load_weights('models/weights')
  print('Model restored.')

def remove_special(data):
  for index, row in enumerate(data):
    for token in row:
      token_index = row.index(token)
      # replacing special characters with an empty string.
      token = re.sub(r'[^A-Za-z0-9]+', '', token)
      data[index][token_index] = token

  return data

def remove_empty(data):
  for index, row in enumerate(data):
    for token in row:
      if not token:  
        # removing empty strings from the list.
        data[index] = list(filter(None, row))

  return data

with open('data/queries.txt', 'r') as f:
    queries_file = f.readlines()

# removing newline characters.
queries_file = [[line.strip()] for line in queries_file]
# tokenizing the queries.
queries = [token.split() for line in queries_file for token in line]
queries = remove_special(queries)
queries = remove_empty(queries)

# applying encoding to the queries.
encoded_queries, _ = encode(queries, [], input_encoder, None)
# converting the queries to numpy.
encoded_queries, _ = to_numpy(encoded_queries, [])
# creating a constant tensor with the encoded queries.
queries_set = tf.constant(encoded_queries)

# creating the input padding mask.
padding_mask_inp = 1 - tf.cast(tf.equal(queries_set, 0), dtype=tf.float32)
padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]

_, query_representations, _ = matching_network(queries_set, queries_set, padding_mask_inp, padding_mask_inp, False)

function_representations = []

for tar in corpus_dataset:
  inp = tf.ones_like(tar)

  # creating the input padding mask.
  padding_mask_inp = 1 - tf.cast(tf.equal(inp, 0), dtype=tf.float32)
  padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]
  # creating the target padding mask.
  padding_mask_tar = 1 - tf.cast(tf.equal(tar, 0), dtype=tf.float32)
  padding_mask_tar = padding_mask_tar[:, tf.newaxis, tf.newaxis, :]

  _, _, function_vectors = matching_network(inp, tar, padding_mask_inp, padding_mask_tar, False)

  # storing the function vectors in a list.
  function_representations.append(function_vectors)

# concatenating all vectors on the first axis.
function_representations = tf.concat(function_representations, axis=-2)

indices = AnnoyIndex(tf.shape(function_representations)[-1], 'euclidean')

for index, vector in enumerate(function_representations):
  indices.add_item(index, vector)

indices.build(10)
indices.save('models/functions.ann')

def get_predictions(vector, indices):
  function_index, distance = indices.get_nns_by_vector(vector, n=100, include_distances=True)

  return function_index, distance

predictions = []

for query_index, vector in enumerate(query_representations):
  function_index, distance = get_predictions(vector, indices)
  
  for index in function_index:
    predictions.append([queries_file[query_index][0], 'java', functions.url[index]])

predictions_dataframe = pd.DataFrame(predictions, columns=['query', 'language', 'url'])
predictions_dataframe.to_csv('data/model_predictions.csv', index=False)