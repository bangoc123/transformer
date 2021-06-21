from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.decoder_layer import DecoderLayer


d_model = 512
h = 8
n = 6
batch_size = 64
vocab_size = 8000
q_length = 100
d_k = 64
d_ff = 2048
activation='relu'
q_length = 100
k_length = v_length = 200
d_model = 512

decoder_layer = DecoderLayer(
    h, d_model, d_ff, activation, 
)

class EncoderLayerTest(tf.test.TestCase):
  def test_call(self):
    q = tf.ones((batch_size, q_length, d_model))
    k = tf.ones((batch_size, k_length, d_model))
    out, self_attn_weights, global_attn_weights = decoder_layer(q, k, None)

    self.assertEqual(
      out.shape,
      (batch_size, q_length, d_model)
    )

    self.assertEqual(
      self_attn_weights.shape,
      (batch_size, h, q_length, q_length)
    )

    self.assertEqual(
      global_attn_weights.shape,
      (batch_size, h, q_length, k_length)
    )

