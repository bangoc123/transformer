from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.encoder import Encoder


d_model = 512
h = 8
n = 6
batch_size = 64
vocab_size = 8000
q_length = 100
d_k = 64
d_ff = 2048
activation='relu'

encoder = Encoder(
    n, h, vocab_size, d_model, d_ff, activation, 
)

q_length = 100
d_model = 512


class EncoderTest(tf.test.TestCase):
  def test_call(self):
    x = tf.ones((batch_size, q_length))
    out = encoder(x, True, None)
    self.assertEqual(
      out.shape,
      (batch_size, q_length, d_model)
    )

