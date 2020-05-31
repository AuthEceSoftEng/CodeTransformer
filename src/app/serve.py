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
from annoy import AnnoyIndex

def api():
    # loading the docstring vocabulary.
    docstring_vocab =  pickle.load(open('data/docstring_vocab.pkl', 'rb'))

    # loading the functions, function tokens and URLs of the whole corpus.
    functions = pd.read_pickle('data/functions.pkl')

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

    class PositionalEncoding(tf.keras.layers.Layer):

        def __init__(self, position, d_model):
            super(PositionalEncoding, self).__init__()

            # creating the positional encoding matrix.
            self.pe = self.positional_encoding(position, d_model)

        def positional_encoding(self, position, d_model):
            # storing word positions to a matrix.
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis]
            # storing embedding components to a matrix.
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

            # calculating the angles.
            angle = tf.multiply(position, 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, dtype=tf.float32)))

            # applying sine to the angles of even indices.
            sines = tf.sin(angle[:, 0::2])
            # applying cosine to the angles of odd indices.
            cosines = tf.cos(angle[:, 1::2])

            # concatenating sines and cosines in one matrix.
            pe = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]

            return tf.cast(pe, dtype=tf.float32)

        def call(self, x):
            # adding positional encoding to the input embeddings on call.
            return x + self.pe[:, :tf.shape(x)[-2], :]

    class MultiHeadAttention(tf.keras.layers.Layer):

        def __init__(self, num_heads, d_model):
            super(MultiHeadAttention, self).__init__()
            
            self.num_heads = num_heads
            self.d_model = d_model
            
            self.head_size = d_model // num_heads

            # creating the weight matrices for each head.
            self.wq = tf.keras.layers.Dense(d_model)
            self.wk = tf.keras.layers.Dense(d_model)
            self.wv = tf.keras.layers.Dense(d_model)

            # creating the weight matrix for the output.
            self.dense = tf.keras.layers.Dense(d_model)

        def call(self, query, key, value, mask):
            # storing the batch size.
            batch_size = tf.shape(query)[-3]

            # passing query, key and value as input to the weight matrices.
            query = self.wq(query)
            key = self.wk(key)
            value = self.wv(value)

            # splitting the dense tensors for each head.
            query = tf.reshape(query, [batch_size, -1, self.num_heads, self.head_size])
            key = tf.reshape(key, [batch_size, -1, self.num_heads, self.head_size])
            value = tf.reshape(value, [batch_size, -1, self.num_heads, self.head_size])

            # transposing the number of heads and sequence length columns.
            query = tf.transpose(query, perm=[0, 2, 1, 3])
            key = tf.transpose(key, perm=[0, 2, 1, 3])
            value = tf.transpose(value, perm=[0, 2, 1, 3])

            # calculating the similarity score.
            query_keyT = tf.matmul(query, key, transpose_b=True)

            # calculating the depth.
            depth = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
            # calculating the scale factor.
            scale = 1 / tf.sqrt(depth)

            # calculating the scaled similarity scores.
            scores = query_keyT * scale

            # masking out key/value pairs.
            if mask is not None:
                scores *= mask
                scores = tf.where(tf.equal(scores, 0), tf.ones_like(scores) * -1e9, scores)

            # calculating the scaled similarity scores' softmax matrix.
            softmax = tf.nn.softmax(scores)

            # calculating the scaled dot-product attention for each head.
            attention = tf.matmul(softmax, value)
            attention = tf.transpose(attention, [0, 2, 1, 3])

            # concatenating the attention heads.
            output = tf.reshape(attention, [batch_size, -1, self.d_model])
            # passing the concatenation as input to a dense layer.
            output = self.dense(output)

            return output

    class FeedForwardNetwork(tf.keras.layers.Layer):

        def __init__(self, dff, d_model):
            super(FeedForwardNetwork, self).__init__()

            # creating the dense layers of the feed forward network.
            self.fc1 = tf.keras.layers.Dense(dff, activation='relu')
            self.fc2 = tf.keras.layers.Dense(d_model)

        def call(self, x):
            # applying the layer with dff units and relu activation.
            fc1 = self.fc1(x)
            # applying the layer with d_model units and no activation.
            output = self.fc2(fc1)

            return output

    class EncoderLayer(tf.keras.layers.Layer):

        def __init__(self, num_heads, dff, d_model, rate):
            super(EncoderLayer, self).__init__()

            # creating the MHA and FFN layers.
            self.mha = MultiHeadAttention(num_heads, d_model)
            self.ffn = FeedForwardNetwork(dff, d_model)

            # creating the dropout layers.
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

            # creating the normalization layers.
            self.normalization1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.normalization2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        def call(self, x, padding_mask, training):
            # applying multi-head attention.
            attention = self.mha(x, x, x, padding_mask)
            dropout1 = self.dropout1(attention, training=training)
            normalization1 = self.normalization1(x + dropout1)

            # applying the feed forward network.
            ffn = self.ffn(normalization1)
            dropout2 = self.dropout2(ffn, training=training)
            output = self.normalization2(normalization1 + dropout2)

            return output

    class Encoder(tf.keras.layers.Layer):

        def __init__(self, num_layers, vocab_size, position, num_heads, dff, d_model, rate):
            super(Encoder, self).__init__()

            self.num_layers = num_layers
            self.d_model = d_model

            # creating the embedding and positional encoding layers.
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
            self.pe = PositionalEncoding(position, d_model)

            # creating the dropout layer.    
            self.dropout = tf.keras.layers.Dropout(rate)

            # creating the encoder layers.    
            self.encoder_layers = [EncoderLayer(num_heads, dff, d_model, rate) for index in range(num_layers)]

        def call(self, x, padding_mask, training):
            # calculating the embeddings and applying the positional encoding.
            x = self.embedding(x)
            x *= tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
            x = self.pe(x)
            x = self.dropout(x, training=training)

            for index in range(self.num_layers):
                # stacking the encoder layers.
                x = self.encoder_layers[index](x, padding_mask, training)

            return x

    class TSSS(tf.keras.layers.Layer):
  
        def __init__(self):
            super(TSSS, self).__init__()

        def call(self, x, y):
            # calculating the theta angle.
            x_normalized = tf.math.l2_normalize(x, axis=-1)
            y_normalized = tf.math.l2_normalize(y, axis=-1)
            theta = tf.acos(tf.matmul(x_normalized, y_normalized, transpose_b=True)) + tf.constant(m.radians(10))

            # calculating the euclidean distance.
            x_matrix = tf.reshape(tf.repeat(x, repeats=tf.shape(x)[-2], axis=-2), shape=[-1, tf.shape(x)[-2], tf.shape(x)[-1]])
            y_matrix = tf.reshape(tf.tile(y, multiples=[tf.shape(y)[-2], 1]), shape=[-1, tf.shape(y)[-2], tf.shape(y)[-1]])
            ed = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_matrix, y_matrix)), axis=-1))

            # calculating the magnitude difference.
            x_sqrt = tf.sqrt(tf.reduce_sum(tf.square(x_matrix), axis=-1))
            y_sqrt = tf.sqrt(tf.reduce_sum(tf.square(y_matrix), axis=-1))
            md = tf.abs(x_sqrt - y_sqrt)

            # calculating the Triangle's Area Similarity.
            x_norm = tf.norm(x, ord='euclidean', axis=-1)[:, tf.newaxis]
            y_norm = tf.norm(y, ord='euclidean', axis=-1)[:, tf.newaxis]
            ts = (tf.matmul(x_norm, y_norm, transpose_b=True) * tf.sin(theta * (tf.constant(m.pi) / 180))) / 2

            # calculating the Sector's Area Similarity.
            ss = (tf.constant(m.pi) * tf.pow((ed + md), 2) * theta) / 360

            # calculating the TS-SS.
            output = ts * ss

            return output

    class MatchingNetwork(tf.keras.Model):

        def __init__(self, num_layers, input_vocab_size, target_vocab_size, input_position, target_position, num_heads, dff, d_model, rate):
            super(MatchingNetwork, self).__init__()

            # creating the Encoders.
            self.encoder1 = Encoder(num_layers, input_vocab_size, input_position, num_heads, dff, d_model, rate)
            self.encoder2 = Encoder(num_layers, target_vocab_size, target_position, num_heads, dff, d_model, rate)

            # creating the Triangle Sector Similarity layer.
            self.similarity = TSSS()

        def call(self, x, y, padding_mask_x, padding_mask_y, training):
            # creating the encoded input padding mask.
            mask1 = tf.squeeze(padding_mask_x)[:, :, tf.newaxis]
            mask1 = -1e9 * (1 - mask1)

            # passing the input data to its corresponding encoder.
            encoded1 = self.encoder1(x, padding_mask_x, training)
            encoded1 = encoded1 + mask1
            encoded1 = tf.reduce_max(encoded1, axis=-2)

            # creating the encoded target padding mask.
            mask2 = tf.squeeze(padding_mask_y)[:, :, tf.newaxis]
            mask2 = -1e9 * (1 - mask2)

            # passing the target data to its corresponding encoder.
            encoded2 = self.encoder2(y, padding_mask_y, training)
            encoded2 = encoded2 + mask2
            encoded2 = tf.reduce_max(encoded2, axis=-2)

            # calculating the similarity and its invert.
            similarity = self.similarity(encoded1, encoded2)
            similarity = 1 / similarity

            return similarity, encoded1, encoded2

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
            predictions.append(functions.url[index])
            predictions.append(functions.function[index].replace('\n', '<br>').replace(' ', '&nbsp;').replace('\t', '&emsp;'))

        return predictions

    return predict