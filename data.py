import io
import numpy as np
import unicodedata
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split

class NMTDataset:
  def __init__(self, inp_lang, targ_lang):
    self.inp_lang = inp_lang
    self.targ_lang = targ_lang
    self.inp_tokenizer = None
    self.target_tokenizer = None

  def preprocess_sentence(self, w, max_length):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()

    # Truncate Length up to ideal_length
    w = " ".join(w.split()[:max_length+1])
    # Add start and end token 
    w = '<start> ' + w + ' <end>'
    return w

  def tokenize(self, lang, max_length):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    # Padding

    tensor = lang_tokenizer.texts_to_sequences(lang)
    # print('---------->', tensor, lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_length)
    # print('---------->', tensor, lang)
    return tensor, lang_tokenizer


  def load_dataset(self, inp_path, targ_path, max_length, num_examples):
    inp_lines = io.open(inp_path, encoding='UTF-8').read().strip().split('\n')[:num_examples]
    targ_lines = io.open(targ_path, encoding='UTF-8').read().strip().split('\n')[:num_examples]
    
    inp_lines = [self.preprocess_sentence(inp, max_length) for inp in inp_lines]
    targ_lines = [self.preprocess_sentence(targ, max_length) for targ in targ_lines]
    
    # Tokenizing
    inp_tensor, inp_tokenizer = self.tokenize(inp_lines, max_length)
    targ_tensor, targ_tokenizer = self.tokenize(targ_lines, max_length)

    # print(inp_tensor)
    
    return inp_tensor, targ_tensor, inp_tokenizer, targ_tokenizer

  def build_dataset(self, inp_path, targ_path, buffer_size, batch_size, max_length, num_examples):

    inp_tensor, targ_tensor, self.inp_tokenizer, self.targ_tokenizer = self.load_dataset(inp_path, targ_path, max_length, num_examples)

    inp_tensor_train, inp_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(inp_tensor, targ_tensor, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_train, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_train, dtype=tf.int64)))

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_val, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_val, dtype=tf.int64)))

    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)

    return train_dataset, val_dataset


