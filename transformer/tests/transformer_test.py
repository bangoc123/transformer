from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.model import Transformer


d_model = 512
h = 8
n = 6
batch_size = 64
inp_vocab_size = 8000
out_vocab_size = 10000
encoder_in_length = 100
decoder_in_length = 200
d_k = 64
d_ff = 2048
activation='relu'

encoder_in = tf.ones((batch_size, encoder_in_length))
decoder_in = tf.ones((batch_size, decoder_in_length))

transfomer = Transformer(
    n, h, inp_vocab_size, out_vocab_size, d_model, d_ff, activation, 
)


class TransformerTest(tf.test.TestCase):
  def test_call(self):
    inp = (encoder_in, decoder_in, True, None, None, None)
    out = transfomer(inp, True)
    self.assertEqual(
      out.shape,
      (batch_size, decoder_in_length, out_vocab_size)
    )

