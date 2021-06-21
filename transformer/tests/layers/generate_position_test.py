from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.generate_position import generate_positional_encoding

q_length = 100
d_model = 512

class PositionEndoingTest(tf.test.TestCase):
  def test_generate_positional_encoding(self):
    
    pos_encoding = generate_positional_encoding(q_length, d_model)

    self.assertEqual(
      pos_encoding.shape,
      (1, q_length, d_model)
    )

