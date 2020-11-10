from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import os
import re
import time
import datetime
import pickle
import math as m
import pandas as pd
import numpy as np
from model import *

# loading the training dataset.
train_dataset = pd.read_pickle('data/train_dataset.pkl')
# loading the validation dataset.
valid_dataset = pd.read_pickle('data/valid_dataset.pkl')
# loading the test dataset.
test_dataset = pd.read_pickle('data/test_dataset.pkl')

# loading the docstring vocabulary.
docstring_vocab =  pickle.load(open('data/docstring_vocab.pkl', 'rb'))
# loading the code vocabulary.
code_vocab =  pickle.load(open('data/code_vocab.pkl', 'rb'))

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
target_encoder = tfds.features.text.TokenTextEncoder(code_vocab)

# loading the input and target data of the training dataset.
training_input = train_dataset['docstring_tokens'].copy(deep=True)
training_target = train_dataset['code_tokens'].copy(deep=True)
# loading the input and target data of the validation dataset.
validation_input = valid_dataset['docstring_tokens'].copy(deep=True)
validation_target = valid_dataset['code_tokens'].copy(deep=True)
# loading the input and target data of the test dataset.
test_input = test_dataset['docstring_tokens'].copy(deep=True)
test_target = test_dataset['code_tokens'].copy(deep=True)

# applying encoding to the input and target data.
encoded_training_input, encoded_training_target = encode(training_input, training_target, input_encoder, target_encoder)
encoded_validation_input, encoded_validation_target = encode(validation_input, validation_target, input_encoder, target_encoder)
encoded_test_input, encoded_test_target = encode(test_input, test_target, input_encoder, target_encoder)
# converting the input and target data to numpy.
encoded_training_input, encoded_training_target = to_numpy(encoded_training_input, encoded_training_target)
encoded_validation_input, encoded_validation_target = to_numpy(encoded_validation_input, encoded_validation_target)
encoded_test_input, encoded_test_target = to_numpy(encoded_test_input, encoded_test_target)

BUFFER_SIZE = 500000

# creating a tensor dataset with the training data.
train_dataset = tf.data.Dataset.from_tensor_slices((encoded_training_input, encoded_training_target))
# caching the dataset for performance optimizations.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(128)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# creating a tensor dataset with the validation data.
valid_dataset = tf.data.Dataset.from_tensor_slices((encoded_validation_input, encoded_validation_target))
# caching the dataset for performance optimizations.
valid_dataset = valid_dataset.cache()
valid_dataset = valid_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
valid_dataset = valid_dataset.batch(1000)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# creating a tensor dataset with the test data.
test_dataset = tf.data.Dataset.from_tensor_slices((encoded_test_input, encoded_test_target))
# caching the dataset for performance optimizations.
test_dataset = test_dataset.cache()
test_dataset = test_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
test_dataset = test_dataset.batch(1000)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

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

# finding the best learning rate using Exponential Decay.
#learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-10, decay_steps=100, decay_rate=1.1, staircase=True)

# creating the Adam optimizer.
optimizer = tf.keras.optimizers.Adam(3.2e-4)

def SquaredMarginLoss(predictions, margin=5):
  # calculating the positive loss.
  positive_loss = tf.linalg.diag_part(predictions)
  positive_loss = tf.pow(tf.maximum(0., margin - positive_loss), 2)
  
  # calculating the negative loss. 
  diag_minus_infinity = tf.linalg.diag(tf.fill(dims=[tf.shape(predictions)[0]], value=-1e9))
  negative_loss = tf.nn.relu(predictions + diag_minus_infinity)
  negative_loss = tf.pow(negative_loss, 2)
  negative_loss = tf.reduce_sum(negative_loss, axis=-1)

  # summing both losses.
  total_loss = positive_loss + negative_loss

  return total_loss

# training metrics.
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
# validation metrics.
valid_loss = tf.keras.metrics.Mean()
valid_accuracy = tf.keras.metrics.CategoricalAccuracy()

def MRR(predictions):
  # getting the correct predictions.
  positive_scores = tf.linalg.diag_part(predictions)
  # calculating their position in respect to the other predictions.
  compared_scores = predictions >= tf.expand_dims(positive_scores, axis=-1)
  # calculating the MRR metric.
  mrr = 1 / tf.reduce_sum(tf.cast(compared_scores, dtype=tf.float32), axis=-1)

  return mrr

@tf.function
def train_step(inp, tar):
  # creating the input padding mask.
  padding_mask_inp = 1 - tf.cast(tf.equal(inp, 0), dtype=tf.float32)
  padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]
  # creating the target padding mask.
  padding_mask_tar = 1 - tf.cast(tf.equal(tar, 0), dtype=tf.float32)
  padding_mask_tar = padding_mask_tar[:, tf.newaxis, tf.newaxis, :]

  # creating the ground truth for the accuracy metric.
  diagonal = tf.ones(tf.shape(inp)[-2])
  one_hot_y = tf.linalg.tensor_diag(diagonal)

  with tf.GradientTape() as tape:
    predictions, en1, en2 = matching_network(inp, tar, padding_mask_inp, padding_mask_tar, training=True)

    loss = SquaredMarginLoss(predictions)
  
  # calculating and applying the gradients.
  gradients = tape.gradient(loss, matching_network.trainable_variables)
  gradients, _ = tf.clip_by_global_norm(gradients, 1)
  optimizer.apply_gradients(zip(gradients, matching_network.trainable_variables))
  
  # calculating the metrics.
  train_loss(loss)
  train_accuracy(one_hot_y, predictions)
  train_mrr = MRR(predictions)
  
  return train_mrr

@tf.function
def valid_step(inp, tar):
  # creating the input padding mask.
  padding_mask_inp = 1 - tf.cast(tf.equal(inp, 0), dtype=tf.float32)
  padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]
  # creating the target padding mask.
  padding_mask_tar = 1 - tf.cast(tf.equal(tar, 0), dtype=tf.float32)
  padding_mask_tar = padding_mask_tar[:, tf.newaxis, tf.newaxis, :]

  # creating the ground truth for the accuracy metric.
  diagonal = tf.ones(tf.shape(inp)[-2])
  one_hot_y = tf.linalg.tensor_diag(diagonal)

  with tf.GradientTape() as tape:
    predictions, en1, en2 = matching_network(inp, tar, padding_mask_inp, padding_mask_tar, training=False)

    loss = SquaredMarginLoss(predictions)
  
  # calculating the metrics.
  valid_loss(loss)
  valid_accuracy(one_hot_y, predictions)
  valid_mrr = MRR(predictions)

  return valid_mrr

@tf.function
def test_step(inp, tar):
  # creating the input padding mask.
  padding_mask_inp = 1 - tf.cast(tf.equal(inp, 0), dtype=tf.float32)
  padding_mask_inp = padding_mask_inp[:, tf.newaxis, tf.newaxis, :]
  # creating the target padding mask.
  padding_mask_tar = 1 - tf.cast(tf.equal(tar, 0), dtype=tf.float32)
  padding_mask_tar = padding_mask_tar[:, tf.newaxis, tf.newaxis, :]

  # creating the ground truth for the accuracy metric.
  diagonal = tf.ones(tf.shape(inp)[-2])
  one_hot_y = tf.linalg.tensor_diag(diagonal)

  with tf.GradientTape() as tape:
    predictions, en1, en2 = matching_network(inp, tar, padding_mask_inp, padding_mask_tar, training=False)

  # calculating the metrics.
  test_mrr = MRR(predictions)

  return test_mrr

summary_writer = tf.summary.create_file_writer('logs/gradient_tape/')

stopwatch = []
step = 0
best_valid_mrr = 0

for epoch in range(40):
  # initializing the timer.
  start = time.time()

  # initializing the training metric storing lists.
  epoch_train_loss = []
  epoch_train_accuracy = []
  epoch_train_mrr = []

  # initializing the validation metric storing lists.
  epoch_valid_loss = []
  epoch_valid_accuracy = []
  epoch_valid_mrr = []
  
  for batch, (inp, tar) in enumerate(train_dataset):
    # resetting the loss and accuracy states for every training batch.
    train_loss.reset_states()
    train_accuracy.reset_states()

    # calling the training step function and storing the metrics.
    epoch_train_mrr.append(tf.reduce_mean(train_step(inp, tar)))
    epoch_train_loss.append(train_loss.result())
    epoch_train_accuracy.append(train_accuracy.result())

    # outputting the training metrics every 1000 batches.
    if batch % 500 == 0:
      print('Epoch {} Batch {} Train Loss {:.10f} Train Accuracy {:.10f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    # logging the training metrics every 100 batches.
    if batch % 100 == 0:
      with summary_writer.as_default():
        tf.summary.scalar('train_loss', data=train_loss.result(), step=step)
        tf.summary.scalar('train_accuracy', data=train_accuracy.result(), step=step)
        #tf.summary.scalar('learning_rate', data=learning_rate(step), step=step)

    step += 1

  for inp, tar in valid_dataset:
    # resetting the loss and accuracy states for every validation batch.
    valid_loss.reset_states()
    valid_accuracy.reset_states()

    # calling the validation step function and storing the metrics.
    epoch_valid_mrr.append(tf.reduce_mean(valid_step(inp, tar)))
    epoch_valid_loss.append(valid_loss.result())
    epoch_valid_accuracy.append(valid_accuracy.result())

  # logging the average training and validation metrics every epoch.
  with summary_writer.as_default():
    tf.summary.scalar('epoch_train_loss', data=tf.reduce_mean(epoch_train_loss), step=epoch)
    tf.summary.scalar('epoch_train_accuracy', data=tf.reduce_mean(epoch_train_accuracy), step=epoch)
    tf.summary.scalar('epoch_train_mrr', data=tf.reduce_mean(epoch_train_mrr), step=epoch)
    tf.summary.scalar('epoch_valid_loss', data=tf.reduce_mean(epoch_valid_loss), step=epoch)
    tf.summary.scalar('epoch_valid_accuracy', data=tf.reduce_mean(epoch_valid_accuracy), step=epoch)
    tf.summary.scalar('epoch_valid_mrr', data=tf.reduce_mean(epoch_valid_mrr), step=epoch)

  # outputting the average training metrics every epoch.
  print('Epoch {} Train Loss {:.10f} Train Accuracy {:.10f} Train MRR {:.10f}'.format(epoch + 1, 
                                                tf.reduce_mean(epoch_train_loss), 
                                                tf.reduce_mean(epoch_train_accuracy),
                                                tf.reduce_mean(epoch_train_mrr)))
  
  # outputting the average validation metrics every epoch.
  print('Epoch {} Valid Loss {:.10f} Valid Accuracy {:.10f} Valid MRR {:.10f}'. format(epoch + 1,
                                                tf.reduce_mean(epoch_valid_loss), 
                                                tf.reduce_mean(epoch_valid_accuracy),
                                                tf.reduce_mean(epoch_valid_mrr)))

  # outputting the epoch time.
  print('Time taken for 1 epoch: {} seconds\n'.format(time.time() - start))
  stopwatch.append(time.time() - start)

  # saving the best model weights.
  if tf.reduce_mean(epoch_valid_mrr) > best_valid_mrr:
    best_valid_mrr = tf.reduce_mean(epoch_valid_mrr)

    matching_network.save_weights('models/weights', overwrite=True)
    print('Model saved at epoch {}\n'.format(epoch+1))

# outputting the total training time.
print('Total training time: {} seconds\n'.format(tf.reduce_sum(stopwatch)))

if os.path.isfile('models/weights.index'):
  matching_network.load_weights('models/weights')
  print('Model restored.')

test_mrr = []

for inp, tar in test_dataset:
  test_mrr.append(tf.reduce_mean(test_step(inp, tar)))

print('Test MRR: {:.10f}'.format(tf.reduce_mean(test_mrr)))