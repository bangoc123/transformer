from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from transformer.layers.generate_mask import generate_padding_mask, generate_look_ahead_mask

q_length = 100
d_model = 512

class MaskGeneratingTest(tf.test.TestCase):
    def test_generate_padding_mask(self):
        inp = tf.constant([[7, 6, 0], [1, 2, 3], [0, 0, 0]])
        exp = tf.constant([[[[0., 0., 1.]]], [[[0., 0., 0.]]], [[[1., 1., 1.]]]], dtype=tf.float32)

        padding_mask = generate_padding_mask(inp)

        self.assertAllEqual(
            padding_mask,
            exp
        )

    def test_generate_look_ahead_mask(self):
        inp = tf.random.uniform((1, 3))
        exp = tf.constant([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]])
        look_ahead_mask = generate_look_ahead_mask(inp.shape[1])
        self.assertAllEqual(
            look_ahead_mask,
            exp
        )
