from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import re
import json
import pickle
import math as m
import pandas as pd
import numpy as np
from model import *
from annoy import AnnoyIndex

def api():
    # loading the docstring vocabulary.
    docstring_vocab =  pickle.load(open('data/docstring_vocab.pkl', 'rb'))

    # loading the functions and URLs of the whole corpus.
    functions_urls = pd.read_pickle('data/functions_urls.pkl')

    # building input_encoder.
    input_encoder = tfds.features.text.TokenTextEncoder(docstring_vocab)

    def encode(inp, input_encoder):
        # encoding input data.
        for index, row in enumerate(inp):
            inp[index] = [input_encoder.encode(token)[0] for token in row]

        return inp

    def to_numpy(inp):
        # converting input data to numpy.
        inp = pd.DataFrame(list(inp))
        inp = inp.to_numpy()
        inp = np.nan_to_num(inp)
        inp = inp.astype(int)

        return inp

    def remove_special(inp):
        for index, row in enumerate(inp):
            for token in row:
                token_index = row.index(token)
                # replacing special characters with an empty string.
                token = re.sub(r'[^A-Za-z0-9]+', '', token)
                inp[index][token_index] = token

        return inp

    def seperate_strings(inp):
        for index, row in enumerate(inp):
            for token in row:
                # if the string is in camelCase format.
                if re.findall(r'[A-Z][a-z][^A-Z]*', token):
                    token_index = row.index(token)
                    # capitalizing the first letter of the token.
                    token = token[0].capitalize() + token[1:]
                    token = re.findall(r'[A-Z][a-z][^A-Z]*|[A-Z]*(?![a-z])|[A-Z][a-z][^A-Z]*', token)
                    # replacing token with an empty string.
                    inp[index][token_index] = ''
                    # adding the seperated words to the list preserving their original position.
                    inp[index] = inp[index][:token_index] + token + inp[index][token_index:]
                    # updating row.
                    row = inp[index]

        return inp

    def remove_empty(inp):
        for index, row in enumerate(inp):
            for token in row:
                if not token:  
                    # removing empty strings from the list.
                    inp[index] = list(filter(None, row))

        return inp

    def lowercase(inp):
        for index, row in enumerate(inp):
            for token in row:
                token_index = row.index(token)
                token = token.lower()
                inp[index][token_index] = token

        return inp

    NUM_LAYERS = 3
    INPUT_VOCAB_SIZE = input_encoder.vocab_size
    TARGET_VOCAB_SIZE = input_encoder.vocab_size
    INPUT_POSITION = input_encoder.vocab_size
    TARGET_POSITION = input_encoder.vocab_size
    NUM_HEADS = 8
    DFF = 512
    D_MODEL = 128
    RATE = 0.1

    matching_network = MatchingNetwork(NUM_LAYERS, INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                                    INPUT_POSITION, TARGET_POSITION, NUM_HEADS,
                                    DFF, D_MODEL, RATE)

    matching_network.load_weights('models/weights')

    indices = AnnoyIndex(D_MODEL, 'euclidean')
    indices.load('models/functions.ann')

    def get_predictions(vector, indices):
        function_index, distance = indices.get_nns_by_vector(vector, n=10, search_k = 1000, include_distances=True)

        return function_index, distance

    def predict(inputData):
        predictions = []

        # tokenizing the query.
        query = [inputData.split(), ['token', 'token']]
        # preprocessing the query.
        query = remove_special(query)
        query = seperate_strings(query)
        query = remove_empty(query)
        query = lowercase(query)

        # applying encoding to the query.
        query = encode(query, input_encoder)
        # converting the query to numpy.
        query = to_numpy(query)
        # creating a constant tensor with the encoded query.
        query = tf.constant(query)

        # creating the input padding mask.
        padding_mask_inp = 1 - tf.cast(tf.equal(query, 0), dtype=tf.float32)
        padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]

        _, query_representations, _ = matching_network(query, query, padding_mask_inp, padding_mask_inp, False)

        function_index, distance = get_predictions(query_representations[0], indices)
            
        for index in function_index:
            predictions.append(functions_urls.url[index])
            predictions.append(functions_urls.function[index].replace('\n', '<br>').replace(' ', '&nbsp;').replace('\t', '&emsp;'))

        return predictions

    return predict