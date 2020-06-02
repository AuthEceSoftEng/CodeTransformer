import re
import wget
import pickle
import collections
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

corpus = pickle.load(open('/data/dataset/java_dedupe_definitions_v2.pkl', 'rb'))
corpus = pd.DataFrame(corpus)

# removing rows with no docstrings.
data = corpus[corpus['docstring_tokens'].map(lambda d: len(d)) > 0]
# resetting DataFrame indices.
data = data.reset_index(drop=True)

# selecting only the first 50,000 rows for faster testing.
#data = data[:50000]

def remove_after_dot(data):
  for index, row in data.iteritems():
    for token in row:
      if token == '.':
        token_index = row.index(token)
        data[index] = row[:token_index]
        break
  
  return data

def remove_non_ascii(data):
  for index, row in data.iteritems():
    for token in row:
      token_index = row.index(token)
      # replacing non-ASCII characters with an empty string.
      token = re.sub(r'[^\x00-\x7f]', '', token)
      data[index][token_index] = token
      
  return data

def remove_special(data):
  for index, row in data.iteritems():
    for token in row:
      token_index = row.index(token)
      # replacing special characters with an empty string.
      token = re.sub(r'[^A-Za-z0-9]+', '', token)
      data[index][token_index] = token

  return data

def seperate_strings(data):
  for index, row in data.iteritems():
    for token in row:
      # if the string is in camelCase format.
      if re.findall(r'[A-Z][a-z][^A-Z]*', token):
        token_index = row.index(token)
        # capitalizing the first letter of the token.
        token = token[0].capitalize() + token[1:]
        token = re.findall(r'[A-Z][a-z][^A-Z]*|[A-Z]*(?![a-z])|[A-Z][a-z][^A-Z]*', token)
        # replacing token with an empty string.
        data[index][token_index] = ''
        # adding the seperated words to the list preserving their original position.
        data[index] = data[index][:token_index] + token + data[index][token_index:]
        # updating row.
        row = data[index]

  return data

def remove_empty(data):
  for index, row in data.iteritems():
    for token in row:
      if not token:  
        # removing empty strings from the list.
        data[index] = list(filter(None, row))

  return data

def fill_empty(identifier, data):
  for (index, row), identifier_row in zip(data.iteritems(), identifier):
    if len(row) < 6 or len(row) > 30:
        data[index] = []
    if not data[index]:
      # splitting identifiers on the dots.
      augmented_row = identifier_row.split('.')
      # capitalizing the first letter of the second half of the identifier.
      augmented_row[1] = augmented_row[1][0].capitalize() + augmented_row[1][1:]
      # seperating all identifier words of the second half using their first capital letter.
      data[index] = re.findall(r'[A-Z][a-z][^A-Z]*|[A-Z]*(?![a-z])|[A-Z][a-z][^A-Z]*', augmented_row[1])

  return data

def lowercase(data):
  for index, row in data.iteritems():
    for token in row:
      token_index = row.index(token)
      token = token.lower()
      data[index][token_index] = token

  return data

def remove_unnecessary(data):
  for index, row in data.iteritems():
    for token in row:
      # if the string contains space, double quotes or is a comment.
      if re.findall(r'[ ]', token) or re.findall(r'(")', token) or re.findall(r'(^//)', token) or re.findall(r'(^/\*)', token) or re.findall(r'(^/\*\*)', token):
        token_index = row.index(token)
        # replacing token with an empty string.
        data[index][token_index] = ''
  
  return data

def replace_symbols(data):
  dictionary = {'(': 'openingparenthesis', 
                ')': 'closingparenthesis',
                '[': 'openingbracket', 
                ']': 'closingbracket',
                '{': 'openingbrace', 
                '}': 'closingbrace',
                '+': 'addoperator', 
                '-': 'subtractoperator',
                '*': 'multiplyoperator', 
                '/': 'divideoperator',
                '^': 'poweroperator', 
                '%': 'modulooperator',
                '=': 'assignoperator', 
                '==': 'equaloperator',
                '!=': 'notequaloperator', 
                '>': 'greateroperator',
                '<': 'lessoperator', 
                '>=': 'greaterequaloperator',
                '<=': 'lessequaloperator', 
                '++': 'incrementoperator',
                '--': 'decrementoperator', 
                '!': 'notoperator',
                '@': 'atsign',
                ';': 'semicolon'}

  for index, row in data.iteritems():
    for token in row:
      # if the string contains one or more of the following symbols.
      if re.findall(r'^[()[\]{}<>+\-*/^%=!@;]', token):
        token_index = row.index(token)
        # replacing token with the name of the symbol contained.
        for symbol, name in dictionary.items():
          if token == symbol:
            data[index][token_index] = name
        
  return data

def trim(data):
  for index, row in data.iteritems():
    if len(row) > 100:
      data[index] = row[:100]

  return data

# copying docstring_tokens column.
docstring_tokens = data['docstring_tokens'].copy(deep=True)
# copying identifier column.
identifier = data['identifier'].copy(deep=True)

# applying the preprocessing functions on all docstring tokens.
docstring_tokens = remove_after_dot(docstring_tokens)
docstring_tokens = remove_non_ascii(docstring_tokens)
docstring_tokens = remove_special(docstring_tokens)
docstring_tokens = seperate_strings(docstring_tokens)
docstring_tokens = remove_empty(docstring_tokens)
docstring_tokens = fill_empty(identifier, docstring_tokens)
docstring_tokens = remove_empty(docstring_tokens)
docstring_tokens = lowercase(docstring_tokens)

# copying function_tokens column.
function_tokens = data['function_tokens'].copy(deep=True)

# applying the preprocessing functions on all function tokens.
function_tokens = remove_non_ascii(function_tokens)
function_tokens = seperate_strings(function_tokens)
function_tokens = remove_unnecessary(function_tokens)
function_tokens = replace_symbols(function_tokens)
function_tokens = remove_special(function_tokens)
function_tokens = remove_empty(function_tokens)
function_tokens = trim(function_tokens)
function_tokens = lowercase(function_tokens)

dataset = pd.concat([docstring_tokens, function_tokens], axis=1)
dataset.to_pickle('/data/dataset.pkl')

# vocabulary of the 10,000 most common docstring tokens.
docstring_vocab = list(token for row in docstring_tokens for token in row)
docstring_vocab = collections.Counter(docstring_vocab)
docstring_vocab = dict(docstring_vocab.most_common(10000))
docstring_vocab = list(docstring_vocab.keys())
# vocabulary of the 10,000 most common function tokens.
function_vocab = list(token for row in function_tokens for token in row)
function_vocab = collections.Counter(function_vocab)
function_vocab = dict(function_vocab.most_common(10000))
function_vocab = list(function_vocab.keys())

with open('/data/docstring_vocab.pkl', 'wb') as docstring_vocab_pkl:
    pickle.dump(docstring_vocab, docstring_vocab_pkl, protocol=pickle.HIGHEST_PROTOCOL)

with open('/data/function_vocab.pkl', 'wb') as function_vocab_pkl:
    pickle.dump(function_vocab, function_vocab_pkl, protocol=pickle.HIGHEST_PROTOCOL)

# copying docstring_tokens column.
corpus_function_tokens = corpus['function_tokens'].copy(deep=True)

# applying the preprocessing functions on all function tokens.
corpus_function_tokens = remove_non_ascii(corpus_function_tokens)
corpus_function_tokens = seperate_strings(corpus_function_tokens)
corpus_function_tokens = remove_unnecessary(corpus_function_tokens)
corpus_function_tokens = replace_symbols(corpus_function_tokens)
corpus_function_tokens = remove_special(corpus_function_tokens)
corpus_function_tokens = remove_empty(corpus_function_tokens)
corpus_function_tokens = trim(corpus_function_tokens)
corpus_function_tokens = lowercase(corpus_function_tokens)

functions = pd.concat([corpus.function, corpus_function_tokens, corpus.url], axis=1)
functions.to_pickle('/data/functions.pkl')