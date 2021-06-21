
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.position_wise_feed_forward_network import ffn

d_model = 512
h = 8
batch_size = 64
q_length = 100
k_length = 200 
v_length = 200
d_k = 64
d_v = 64


x = tf.ones((batch_size, q_length, d_model))

class FFFTest(tf.test.TestCase):
  def test_fnn(self):
    network = ffn()
    out = network(x)

    self.assertEqual(
      out.shape,
      (batch_size, q_length, d_model)
    )
