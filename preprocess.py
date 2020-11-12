import re
import wget
import gzip
import shutil
import pickle
import collections
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

path = 'data/dataset/java/final/jsonl/MODE/java_MODE_INDEX.jsonl'
mode = ['train', 'valid', 'test']

for m in mode:
  if m == 'train':
    train_data = pd.DataFrame()

    for i in range(0, 16):
      file_path = path.replace('MODE', m)
      file_path = file_path.replace('INDEX', str(i))

      with gzip.open(file_path + '.gz', 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

      train_data_temp = pd.read_json(file_path, lines=True)
      train_data = train_data.append(train_data_temp)

    # resetting indices.
    train_data = train_data.reset_index(drop=True)

  elif m == 'valid':
    valid_data = pd.DataFrame()

    for i in range(0, 1):
      file_path = path.replace('MODE', m)
      file_path = file_path.replace('INDEX', str(i))

      with gzip.open(file_path + '.gz', 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

      valid_data_temp = pd.read_json(file_path, lines=True)
      valid_data = valid_data.append(valid_data_temp)

  elif m == 'test':
    test_data = pd.DataFrame()

    for i in range(0, 1):
      file_path = path.replace('MODE', m)
      file_path = file_path.replace('INDEX', str(i))

      with gzip.open(file_path + '.gz', 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

      test_data_temp = pd.read_json(file_path, lines=True)
      test_data = test_data.append(test_data_temp)

corpus = pickle.load(open('data/dataset/java_dedupe_definitions_v2.pkl', 'rb'))
corpus = pd.DataFrame(corpus)

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

def fill_empty(function_name, data):
  for (index, row), function_name_row in zip(data.iteritems(), function_name):
    if len(row) < 6 or len(row) > 30:
        data[index] = []
    if not data[index]:
      # splitting function's name on the dots.
      augmented_row = function_name_row.split('.')
      # capitalizing the first letter of the second half of the function's name.
      augmented_row[1] = augmented_row[1][0].capitalize() + augmented_row[1][1:]
      # seperating all function's name words of the second half using their first capital letter.
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

# copying docstring_tokens column of the training dataset.
train_docstring_tokens = train_data['docstring_tokens'].copy(deep=True)
# copying func_name column of the training dataset.
train_function_name = train_data['func_name'].copy(deep=True)

# applying the preprocessing functions on all docstring tokens of the training dataset.
train_docstring_tokens = remove_after_dot(train_docstring_tokens)
train_docstring_tokens = remove_non_ascii(train_docstring_tokens)
train_docstring_tokens = remove_special(train_docstring_tokens)
train_docstring_tokens = seperate_strings(train_docstring_tokens)
train_docstring_tokens = remove_empty(train_docstring_tokens)
train_docstring_tokens = fill_empty(train_function_name, train_docstring_tokens)
train_docstring_tokens = remove_empty(train_docstring_tokens)
train_docstring_tokens = lowercase(train_docstring_tokens)

# copying docstring_tokens column of the validation dataset.
valid_docstring_tokens = valid_data['docstring_tokens'].copy(deep=True)
# copying func_name column of the validation dataset.
valid_function_name = valid_data['func_name'].copy(deep=True)

# applying the preprocessing functions on all docstring tokens of the validation dataset.
valid_docstring_tokens = remove_after_dot(valid_docstring_tokens)
valid_docstring_tokens = remove_non_ascii(valid_docstring_tokens)
valid_docstring_tokens = remove_special(valid_docstring_tokens)
valid_docstring_tokens = seperate_strings(valid_docstring_tokens)
valid_docstring_tokens = remove_empty(valid_docstring_tokens)
valid_docstring_tokens = fill_empty(valid_function_name, valid_docstring_tokens)
valid_docstring_tokens = remove_empty(valid_docstring_tokens)
valid_docstring_tokens = lowercase(valid_docstring_tokens)

# copying docstring_tokens column of the test dataset.
test_docstring_tokens = test_data['docstring_tokens'].copy(deep=True)
# copying func_name column of the test dataset.
test_function_name = test_data['func_name'].copy(deep=True)

# applying the preprocessing functions on all docstring tokens of the test dataset.
test_docstring_tokens = remove_after_dot(test_docstring_tokens)
test_docstring_tokens = remove_non_ascii(test_docstring_tokens)
test_docstring_tokens = remove_special(test_docstring_tokens)
test_docstring_tokens = seperate_strings(test_docstring_tokens)
test_docstring_tokens = remove_empty(test_docstring_tokens)
test_docstring_tokens = fill_empty(test_function_name, test_docstring_tokens)
test_docstring_tokens = remove_empty(test_docstring_tokens)
test_docstring_tokens = lowercase(test_docstring_tokens)

# copying code_tokens column of the training dataset.
train_code_tokens = train_data['code_tokens'].copy(deep=True)

# applying the preprocessing functions on all code tokens of the training dataset.
train_code_tokens = remove_non_ascii(train_code_tokens)
train_code_tokens = seperate_strings(train_code_tokens)
train_code_tokens = remove_unnecessary(train_code_tokens)
train_code_tokens = replace_symbols(train_code_tokens)
train_code_tokens = remove_special(train_code_tokens)
train_code_tokens = remove_empty(train_code_tokens)
train_code_tokens = trim(train_code_tokens)
train_code_tokens = lowercase(train_code_tokens)

# copying code_tokens column of the validation dataset.
valid_code_tokens = valid_data['code_tokens'].copy(deep=True)

# applying the preprocessing functions on all code tokens of the validation dataset.
valid_code_tokens = remove_non_ascii(valid_code_tokens)
valid_code_tokens = seperate_strings(valid_code_tokens)
valid_code_tokens = remove_unnecessary(valid_code_tokens)
valid_code_tokens = replace_symbols(valid_code_tokens)
valid_code_tokens = remove_special(valid_code_tokens)
valid_code_tokens = remove_empty(valid_code_tokens)
valid_code_tokens = trim(valid_code_tokens)
valid_code_tokens = lowercase(valid_code_tokens)

# copying code_tokens column of the test dataset.
test_code_tokens = test_data['code_tokens'].copy(deep=True)

# applying the preprocessing functions on all code tokens of the test dataset.
test_code_tokens = remove_non_ascii(test_code_tokens)
test_code_tokens = seperate_strings(test_code_tokens)
test_code_tokens = remove_unnecessary(test_code_tokens)
test_code_tokens = replace_symbols(test_code_tokens)
test_code_tokens = remove_special(test_code_tokens)
test_code_tokens = remove_empty(test_code_tokens)
test_code_tokens = trim(test_code_tokens)
test_code_tokens = lowercase(test_code_tokens)

train_dataset = pd.concat([train_docstring_tokens, train_code_tokens], axis=1)
train_dataset.to_pickle('data/train_dataset.pkl')

valid_dataset = pd.concat([valid_docstring_tokens, valid_code_tokens], axis=1)
valid_dataset.to_pickle('data/valid_dataset.pkl')

test_dataset = pd.concat([test_docstring_tokens, test_code_tokens], axis=1)
test_dataset.to_pickle('data/test_dataset.pkl')

# vocabulary of the 10,000 most common docstring tokens.
docstring_vocab = list(token for row in train_docstring_tokens for token in row)
docstring_vocab = collections.Counter(docstring_vocab)
docstring_vocab = dict(docstring_vocab.most_common(10000))
docstring_vocab = list(docstring_vocab.keys())
# vocabulary of the 10,000 most common code tokens.
code_vocab = list(token for row in train_code_tokens for token in row)
code_vocab = collections.Counter(code_vocab)
code_vocab = dict(code_vocab.most_common(10000))
code_vocab = list(code_vocab.keys())

with open('data/docstring_vocab.pkl', 'wb') as docstring_vocab_pkl:
  pickle.dump(docstring_vocab, docstring_vocab_pkl, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/code_vocab.pkl', 'wb') as code_vocab_pkl:
  pickle.dump(code_vocab, code_vocab_pkl, protocol=pickle.HIGHEST_PROTOCOL)

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
functions.to_pickle('data/functions.pkl')