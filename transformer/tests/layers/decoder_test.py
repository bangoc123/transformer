from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.decoder import Decoder


d_model = 512
h = 8
n = 6
batch_size = 64
vocab_size = 10000
q_length = 100
d_k = 64
d_ff = 2048
activation='relu'
q_length = 100
k_length = v_length = 200
d_model = 512

decoder = Decoder(
    n, h, vocab_size, d_model, d_ff, activation, 
)

encoder_out = tf.ones((
    batch_size, k_length, d_model
))

q_length = 100
d_model = 512

class DecoderTest(tf.test.TestCase):
  def test_call(self):
    q = tf.ones((batch_size, q_length))
    out, attentionWeights = decoder(q, encoder_out, True, None)
    self.assertEqual(
      out.shape,
      (batch_size, q_length, d_model)
    )

    self.assertEqual(
      attentionWeights['decoder_layer_1_self_attn_weights'].shape,
      (batch_size, h, q_length, q_length)
    )

    self.assertEqual(
      attentionWeights['decoder_layer_1_global_attn_weights'].shape,
      (batch_size, h, q_length, k_length)
    )
